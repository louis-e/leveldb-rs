//! Bulk loading: writes a new database straight from sorted entries.
//!
//! A database that is filled once, from entries whose order is known, does not need the
//! write-ahead log, the memtable or any compaction: its tables can be written directly, each
//! entry compressed exactly once. [`TableWriter`] builds one table file. Tables that cover
//! disjoint key ranges are independent of each other, so they can be built on separate
//! threads, each thread with its own [`Options`] (which are not `Send`).
//! [`install_tables`] then writes the manifest that makes the tables a database.
//!
//! ```
//! use rusty_leveldb::bulk::{install_tables, level_for_size, TableWriter};
//! use rusty_leveldb::{LdbIterator, DB};
//! use std::path::Path;
//!
//! let opt = rusty_leveldb::in_memory();
//! let db = Path::new("bulkdb");
//!
//! // File numbers must be unique within the database and greater than zero.
//! let mut writer = TableWriter::create(&opt, db, 1, 1).unwrap();
//! writer.add(b"apple", b"red").unwrap();
//! writer.add(b"banana", b"yellow").unwrap();
//! let table = writer.finish().unwrap().unwrap();
//!
//! install_tables(&opt, db, level_for_size(table.size as u64), vec![table]).unwrap();
//!
//! let mut db = DB::open(db, opt).unwrap();
//! assert_eq!(b"yellow", &*db.get(b"banana").unwrap());
//! ```

use crate::error::{err, Result, StatusCode};
use crate::key_types::{LookupKey, ValueType};
use crate::log::LogWriter;
use crate::options::Options;
use crate::table_builder::TableBuilder;
use crate::table_cache::table_file_name;
use crate::types::{FileMetaData, FileNum, SequenceNumber, MAX_SEQUENCE_NUMBER, NUM_LEVELS};
use crate::version_edit::VersionEdit;
use crate::version_set::{current_file_name, manifest_file_name, set_current_file};

use std::cmp::Ordering;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Buffer between the table builder and its file, so a block is one write, not three.
const FILE_BUFFER_SIZE: usize = 64 << 10;

/// Largest total size of the files on level 1; each deeper level holds ten times more.
const LEVEL_1_MAX_BYTES: u64 = 10 << 20;

/// A finished table file, as [`install_tables`] needs it. `Send`, unlike the writer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BulkTable {
    /// The number the file is named after.
    pub num: FileNum,
    /// Size of the file in bytes.
    pub size: usize,
    /// Number of entries in the table.
    pub entries: usize,
    /// Highest sequence number of the table's entries.
    pub last_seq: SequenceNumber,
    // Both in internal key format, as the manifest stores them.
    smallest: Vec<u8>,
    largest: Vec<u8>,
}

impl BulkTable {
    /// The first key of the table.
    pub fn smallest_key(&self) -> &[u8] {
        user_key(&self.smallest)
    }

    /// The last key of the table.
    pub fn largest_key(&self) -> &[u8] {
        user_key(&self.largest)
    }
}

/// The user key of an internal key, which ends in an 8 byte tag.
fn user_key(internal_key: &[u8]) -> &[u8] {
    &internal_key[..internal_key.len() - 8]
}

/// Writes one table file of a database that is being bulk loaded.
pub struct TableWriter {
    opt: Options,
    path: PathBuf,
    builder: Option<TableBuilder<BufWriter<Box<dyn Write>>>>,
    num: FileNum,
    next_seq: SequenceNumber,
    entries: usize,
    smallest: Vec<u8>,
    largest: Vec<u8>,
}

impl TableWriter {
    /// Creates the table file numbered `num` in the database directory `db`, which must
    /// exist. Entries get consecutive sequence numbers from `first_seq` on.
    ///
    /// File numbers must be unique within the database and greater than zero. Sequence
    /// numbers need not be unique across tables, since no key occurs twice.
    pub fn create(
        opt: &Options,
        db: &Path,
        num: FileNum,
        first_seq: SequenceNumber,
    ) -> Result<TableWriter> {
        if num == 0 {
            return err(StatusCode::InvalidArgument, "table file number 0");
        }
        if first_seq > MAX_SEQUENCE_NUMBER {
            return err(StatusCode::InvalidArgument, "sequence number out of range");
        }
        let path = table_file_name(db, num);
        let file = opt.env.open_writable_file(&path)?;
        let dst = BufWriter::with_capacity(FILE_BUFFER_SIZE, file);
        Ok(TableWriter {
            opt: opt.clone(),
            path,
            builder: Some(TableBuilder::new(opt.clone(), dst)),
            num,
            next_seq: first_seq,
            entries: 0,
            smallest: Vec::new(),
            largest: Vec::new(),
        })
    }

    /// Adds an entry. Keys must be added in strictly ascending order of the comparator.
    pub fn add(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        if self.entries > 0 && self.opt.cmp.cmp(user_key(&self.largest), key) != Ordering::Less {
            return err(
                StatusCode::InvalidArgument,
                "bulk-loaded keys must be strictly ascending",
            );
        }
        if self.next_seq > MAX_SEQUENCE_NUMBER {
            return err(StatusCode::InvalidArgument, "sequence number out of range");
        }
        let internal = LookupKey::new_full(key, self.next_seq, ValueType::TypeValue);
        let internal = internal.internal_key();
        self.builder
            .as_mut()
            .expect("builder is present until finish")
            .add(internal, value)?;
        if self.entries == 0 {
            self.smallest = internal.to_vec();
        }
        self.largest.clear();
        self.largest.extend_from_slice(internal);
        self.entries += 1;
        self.next_seq += 1;
        Ok(())
    }

    /// Number of entries added so far.
    pub fn entries(&self) -> usize {
        self.entries
    }

    /// Roughly the size the file would have if it were finished now.
    pub fn size_estimate(&self) -> usize {
        self.builder
            .as_ref()
            .expect("builder is present until finish")
            .size_estimate()
    }

    /// Finishes the file. A writer that got no entry leaves no file and returns `None`.
    pub fn finish(mut self) -> Result<Option<BulkTable>> {
        let builder = self
            .builder
            .take()
            .expect("builder is present until finish");
        if self.entries == 0 {
            drop(builder);
            self.opt.env.delete(&self.path)?;
            return Ok(None);
        }
        let size = builder.finish()?;
        Ok(Some(BulkTable {
            num: self.num,
            size,
            entries: self.entries,
            last_seq: self.next_seq - 1,
            smallest: std::mem::take(&mut self.smallest),
            largest: std::mem::take(&mut self.largest),
        }))
    }
}

/// The shallowest level a bulk load of `total_bytes` fits without being compacted further
/// on the first open: level 1 up to 10 MiB, ten times more on each level below it.
pub fn level_for_size(total_bytes: u64) -> usize {
    let mut level = 1;
    let mut max_bytes = LEVEL_1_MAX_BYTES;
    while total_bytes > max_bytes && level < NUM_LEVELS - 1 {
        level += 1;
        max_bytes = max_bytes.saturating_mul(10);
    }
    level
}

/// Makes `tables` the contents of a new database in the directory `db`, all on `level`.
///
/// The tables must cover disjoint key ranges, as the files of every level but the first do.
/// The directory must not hold a database yet. No entries may be passed.
pub fn install_tables(
    opt: &Options,
    db: &Path,
    level: usize,
    mut tables: Vec<BulkTable>,
) -> Result<()> {
    if level >= NUM_LEVELS {
        return err(StatusCode::InvalidArgument, "no such level");
    }
    let env = opt.env.as_ref().as_ref();
    if env.exists(&current_file_name(db))? {
        return err(
            StatusCode::AlreadyExists,
            "the directory already holds a database",
        );
    }

    tables.sort_by(|a, b| opt.cmp.cmp(a.smallest_key(), b.smallest_key()));
    for pair in tables.windows(2) {
        if opt.cmp.cmp(pair[0].largest_key(), pair[1].smallest_key()) != Ordering::Less {
            return err(
                StatusCode::InvalidArgument,
                "bulk-loaded tables must cover disjoint key ranges",
            );
        }
    }
    let mut nums: Vec<FileNum> = tables.iter().map(|t| t.num).collect();
    nums.sort_unstable();
    if nums.windows(2).any(|pair| pair[0] == pair[1]) {
        return err(StatusCode::InvalidArgument, "duplicate table file number");
    }

    let manifest_num = nums.last().map_or(1, |last| last + 1);
    let mut edit = VersionEdit::new();
    edit.set_comparator_name(opt.cmp.id());
    edit.set_log_num(0);
    edit.set_next_file(manifest_num + 1);
    edit.set_last_seq(tables.iter().map(|t| t.last_seq).max().unwrap_or(0));
    for table in tables {
        edit.add_file(
            level,
            FileMetaData {
                allowed_seeks: 0,
                num: table.num,
                size: table.size,
                smallest: table.smallest.into(),
                largest: table.largest.into(),
            },
        );
    }

    {
        let manifest = env.open_writable_file(&manifest_file_name(db, manifest_num))?;
        let mut log = LogWriter::new(manifest);
        log.add_record(&edit.encode())?;
        log.flush()?;
    }
    set_current_file(env, db, manifest_num)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::LdbIterator;
    use crate::{options, DB};

    fn key(i: usize) -> Vec<u8> {
        format!("key{i:06}").into_bytes()
    }

    fn value(i: usize) -> Vec<u8> {
        // Larger than a block now and then, as a bulk-loaded value may well be.
        vec![(i % 251) as u8; 16 + (i % 7) * 1500]
    }

    /// Tables of `per_table` entries each, covering `0..count`.
    fn build_tables(opt: &Options, db: &Path, count: usize, per_table: usize) -> Vec<BulkTable> {
        let mut tables = Vec::new();
        for (t, start) in (0..count).step_by(per_table).enumerate() {
            let mut writer =
                TableWriter::create(opt, db, t as FileNum + 1, start as u64 + 1).unwrap();
            for i in start..(start + per_table).min(count) {
                writer.add(&key(i), &value(i)).unwrap();
            }
            assert!(writer.size_estimate() > 0);
            tables.push(writer.finish().unwrap().unwrap());
        }
        tables
    }

    #[test]
    fn a_bulk_loaded_database_reads_back_whole() {
        let opt = options::for_test();
        let db = Path::new("bulk");
        const COUNT: usize = 2000;
        let mut tables = build_tables(&opt, db, COUNT, 300);
        assert_eq!(tables.len(), 7);
        assert_eq!(tables[0].smallest_key(), key(0));
        assert_eq!(tables[0].largest_key(), key(299));
        assert_eq!(tables[6].last_seq, COUNT as u64);
        let total: u64 = tables.iter().map(|t| t.size as u64).sum();
        // Installed in any order.
        tables.reverse();
        install_tables(&opt, db, level_for_size(total), tables).unwrap();

        let mut db = DB::open(db, opt).unwrap();
        for i in (0..COUNT).step_by(37) {
            assert_eq!(db.get(&key(i)).as_deref(), Some(&value(i)[..]));
        }
        assert!(db.get(b"key").is_none());
        assert!(db.get(&key(COUNT)).is_none());

        let mut iter = db.new_iter().unwrap();
        let mut seen = 0;
        while let Some((k, v)) = iter.next() {
            assert_eq!(k, key(seen));
            assert_eq!(v, value(seen));
            seen += 1;
        }
        assert_eq!(seen, COUNT);
    }

    #[test]
    fn a_bulk_loaded_database_takes_later_writes() {
        let opt = options::for_test();
        let db = Path::new("bulk");
        let tables = build_tables(&opt, db, 500, 100);
        install_tables(&opt, db, 1, tables).unwrap();

        let mut db = DB::open(db, opt).unwrap();
        // A later write must win over the bulk-loaded entry of the same key.
        db.put(&key(7), b"new").unwrap();
        db.delete(&key(8)).unwrap();
        db.flush().unwrap();
        db.compact_range(b"key", b"kez").unwrap();
        assert_eq!(db.get(&key(7)).as_deref(), Some(&b"new"[..]));
        assert!(db.get(&key(8)).is_none());
        assert_eq!(db.get(&key(9)).as_deref(), Some(&value(9)[..]));
    }

    #[test]
    fn an_empty_database_can_be_installed() {
        let opt = options::for_test();
        let db = Path::new("bulk");
        let writer = TableWriter::create(&opt, db, 1, 1).unwrap();
        assert!(writer.finish().unwrap().is_none());
        assert!(!opt.env.exists(&table_file_name(db, 1)).unwrap());
        install_tables(&opt, db, 1, Vec::new()).unwrap();

        let mut db = DB::open(db, opt).unwrap();
        assert!(db.get(b"anything").is_none());
        db.put(b"anything", b"now").unwrap();
        assert_eq!(db.get(b"anything").as_deref(), Some(&b"now"[..]));
    }

    #[test]
    fn keys_out_of_order_are_refused() {
        let opt = options::for_test();
        let db = Path::new("bulk");
        let mut writer = TableWriter::create(&opt, db, 1, 1).unwrap();
        writer.add(b"b", b"1").unwrap();
        assert_eq!(
            writer.add(b"a", b"2").unwrap_err().code,
            StatusCode::InvalidArgument
        );
        assert_eq!(
            writer.add(b"b", b"2").unwrap_err().code,
            StatusCode::InvalidArgument
        );
        writer.add(b"c", b"3").unwrap();
        assert_eq!(writer.entries(), 2);
        assert!(TableWriter::create(&opt, db, 0, 1).is_err());
    }

    #[test]
    fn tables_that_overlap_or_share_a_number_are_refused() {
        let opt = options::for_test();
        let db = Path::new("bulk");
        let table = |num, keys: &[&[u8]]| {
            let mut writer = TableWriter::create(&opt, db, num, 1).unwrap();
            for key in keys {
                writer.add(key, b"v").unwrap();
            }
            writer.finish().unwrap().unwrap()
        };
        let overlapping = vec![table(1, &[b"a", b"c"]), table(2, &[b"b", b"d"])];
        assert_eq!(
            install_tables(&opt, db, 1, overlapping).unwrap_err().code,
            StatusCode::InvalidArgument
        );
        let touching = vec![table(3, &[b"a", b"c"]), table(4, &[b"c", b"d"])];
        assert_eq!(
            install_tables(&opt, db, 1, touching).unwrap_err().code,
            StatusCode::InvalidArgument
        );
        let same_number = vec![table(5, &[b"a"]), table(5, &[b"b"])];
        assert_eq!(
            install_tables(&opt, db, 1, same_number).unwrap_err().code,
            StatusCode::InvalidArgument
        );
        assert_eq!(
            install_tables(&opt, db, NUM_LEVELS, Vec::new())
                .unwrap_err()
                .code,
            StatusCode::InvalidArgument
        );

        install_tables(&opt, db, 1, vec![table(6, &[b"a"])]).unwrap();
        assert_eq!(
            install_tables(&opt, db, 1, vec![table(7, &[b"b"])])
                .unwrap_err()
                .code,
            StatusCode::AlreadyExists
        );
    }

    #[test]
    fn the_level_grows_with_the_size() {
        const MIB: u64 = 1 << 20;
        assert_eq!(level_for_size(0), 1);
        assert_eq!(level_for_size(10 * MIB), 1);
        assert_eq!(level_for_size(10 * MIB + 1), 2);
        assert_eq!(level_for_size(100 * MIB), 2);
        assert_eq!(level_for_size(1000 * MIB), 3);
        assert_eq!(level_for_size(10_000 * MIB + 1), 5);
        assert_eq!(level_for_size(u64::MAX), NUM_LEVELS - 1);
    }
}
