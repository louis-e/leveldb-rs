use crate::key_types::{self, LookupKey};
use crate::types;

use std::cmp::Ordering;
use std::rc::Rc;

type WrappedCmp = Rc<Box<dyn Cmp>>;

/// Comparator trait, supporting types that can be nested (i.e., add additional functionality on
/// top of an inner comparator)
pub trait Cmp {
    /// Compare to byte strings, bytewise.
    fn cmp(&self, a: &[u8], b: &[u8]) -> Ordering;

    /// Return the shortest byte string that compares "Greater" to the first argument and "Less" to
    /// the second one.
    fn find_shortest_sep(&self, from: &[u8], to: &[u8]) -> Vec<u8>;
    /// Return the shortest byte string that compares "Greater" to the argument.
    fn find_short_succ(&self, key: &[u8]) -> Vec<u8>;

    /// A unique identifier for a comparator. A comparator wrapper (like InternalKeyCmp) may
    /// return the id of its inner comparator.
    fn id(&self) -> &'static str;
}

/// The default byte-wise comparator.
#[derive(Clone)]
pub struct DefaultCmp;

impl Cmp for DefaultCmp {
    fn cmp(&self, a: &[u8], b: &[u8]) -> Ordering {
        a.cmp(b)
    }

    fn id(&self) -> &'static str {
        "leveldb.BytewiseComparator"
    }

    fn find_shortest_sep(&self, a: &[u8], b: &[u8]) -> Vec<u8> {
        if a == b {
            return a.to_vec();
        }

        let min = if a.len() < b.len() { a.len() } else { b.len() };
        let mut diff_at = 0;

        while diff_at < min && a[diff_at] == b[diff_at] {
            diff_at += 1;
        }

        // First, try to find a short separator. If that fails, try a backup mechanism below.
        while diff_at < min {
            let diff = a[diff_at];
            if diff < 0xff && diff + 1 < b[diff_at] {
                let mut sep = Vec::from(&a[0..diff_at + 1]);
                sep[diff_at] += 1;
                assert!(self.cmp(&sep, b) == Ordering::Less);
                return sep;
            }

            diff_at += 1;
        }

        let mut sep = Vec::with_capacity(a.len() + 1);
        sep.extend_from_slice(a);
        // Try increasing a and check if it's still smaller than b. First find the last byte
        // smaller than 0xff, and then increment that byte. Only if the separator is lesser than b,
        // return it.
        // The empty key has no byte to increase, and is its own separator from any `b`.
        if a.is_empty() {
            return sep;
        }
        let mut i = a.len() - 1;
        while i > 0 && sep[i] == 0xff {
            i -= 1;
        }
        if sep[i] < 0xff {
            sep[i] += 1;
            if self.cmp(&sep, b) == Ordering::Less {
                return sep;
            } else {
                sep[i] -= 1;
            }
        }

        // Backup case: either `a` is full of 0xff, or all different places are less than 2
        // characters apart.
        // The result is not necessarily short, but a good separator: e.g., "abc" vs "abd" ->
        // "abc\0", which is greater than abc and lesser than abd.
        // Append a 0 byte; by making it longer than a, it will compare greater to it.
        sep.extend_from_slice(&[0]);
        // Unless that is `b` itself ("abc" vs "abc\0"): nothing lies between the two then,
        // and a separator must stay below `b`.
        if sep.as_slice() == b {
            sep.pop();
        }
        sep
    }

    fn find_short_succ(&self, a: &[u8]) -> Vec<u8> {
        let mut result = a.to_vec();
        for i in 0..a.len() {
            if a[i] != 0xff {
                result[i] += 1;
                result.resize(i + 1, 0);
                return result;
            }
        }
        // Rare path
        result.push(255);
        result
    }
}

/// Same as memtable_key_cmp, but for InternalKeys.
#[derive(Clone)]
pub struct InternalKeyCmp(pub Rc<Box<dyn Cmp>>);

impl Cmp for InternalKeyCmp {
    fn cmp(&self, a: &[u8], b: &[u8]) -> Ordering {
        key_types::cmp_internal_key(self.0.as_ref().as_ref(), a, b)
    }

    fn id(&self) -> &'static str {
        self.0.id()
    }

    fn find_shortest_sep(&self, a: &[u8], b: &[u8]) -> Vec<u8> {
        if a == b {
            return a.to_vec();
        }

        let (_, seqa, keya) = key_types::parse_internal_key(a);
        let (_, _, keyb) = key_types::parse_internal_key(b);

        let sep: Vec<u8> = self.0.find_shortest_sep(keya, keyb);

        if sep.len() < keya.len() && self.0.cmp(keya, &sep) == Ordering::Less {
            return LookupKey::new(&sep, types::MAX_SEQUENCE_NUMBER)
                .internal_key()
                .to_vec();
        }
        // A separator that is `b`'s user key would sort after `b` whenever `a` has the lower
        // sequence number, and a lookup of that key would then stop one block early. One
        // that is `a`'s user key is `a` itself, tag and all: rebuilt as a value it would
        // sort before `a` if `a` is a deletion.
        if self.0.cmp(&sep, keyb) != Ordering::Less || self.0.cmp(&sep, keya) != Ordering::Greater {
            return a.to_vec();
        }
        LookupKey::new(&sep, seqa).internal_key().to_vec()
    }

    fn find_short_succ(&self, a: &[u8]) -> Vec<u8> {
        let (_, seq, key) = key_types::parse_internal_key(a);
        let succ: Vec<u8> = self.0.find_short_succ(key);
        LookupKey::new(&succ, seq).internal_key().to_vec()
    }
}

impl InternalKeyCmp {
    /// cmp_inner compares a and b using the underlying comparator (the "user comparator").
    pub fn cmp_inner(&self, a: &[u8], b: &[u8]) -> Ordering {
        self.0.cmp(a, b)
    }
}

/// An internal comparator wrapping a user-supplied comparator. This comparator is used to compare
/// memtable keys, which contain length prefixes and a sequence number.
/// The ordering is determined by asking the wrapped comparator; ties are broken by *reverse*
/// ordering the sequence numbers. (This means that when having an entry abx/4 and seRching for
/// abx/5, then abx/4 is counted as "greater-or-equal", making snapshot functionality work at all)
#[derive(Clone)]
pub struct MemtableKeyCmp(pub Rc<Box<dyn Cmp>>);

impl Cmp for MemtableKeyCmp {
    fn cmp(&self, a: &[u8], b: &[u8]) -> Ordering {
        key_types::cmp_memtable_key(self.0.as_ref().as_ref(), a, b)
    }

    fn id(&self) -> &'static str {
        self.0.id()
    }

    // The following two impls should not be used (by principle) although they should be correct.
    // They will crash the program.
    fn find_shortest_sep(&self, _: &[u8], _: &[u8]) -> Vec<u8> {
        panic!("find* functions are invalid on MemtableKeyCmp");
    }

    fn find_short_succ(&self, _: &[u8]) -> Vec<u8> {
        panic!("find* functions are invalid on MemtableKeyCmp");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use key_types::LookupKey;

    #[test]
    fn test_cmp_defaultcmp_shortest_sep() {
        assert_eq!(
            DefaultCmp.find_shortest_sep("abcd".as_bytes(), "abcf".as_bytes()),
            "abce".as_bytes()
        );
        assert_eq!(
            DefaultCmp.find_shortest_sep("abc".as_bytes(), "acd".as_bytes()),
            "abd".as_bytes()
        );
        assert_eq!(
            DefaultCmp.find_shortest_sep("abcdefghi".as_bytes(), "abcffghi".as_bytes()),
            "abce".as_bytes()
        );
        assert_eq!(
            DefaultCmp.find_shortest_sep("a".as_bytes(), "a".as_bytes()),
            "a".as_bytes()
        );
        assert_eq!(
            DefaultCmp.find_shortest_sep("a".as_bytes(), "b".as_bytes()),
            "a\0".as_bytes()
        );
        assert_eq!(
            DefaultCmp.find_shortest_sep("abc".as_bytes(), "zzz".as_bytes()),
            "b".as_bytes()
        );
        assert_eq!(
            DefaultCmp.find_shortest_sep("yyy".as_bytes(), "z".as_bytes()),
            "yyz".as_bytes()
        );
        assert_eq!(
            DefaultCmp.find_shortest_sep("".as_bytes(), "".as_bytes()),
            "".as_bytes()
        );
    }

    #[test]
    fn test_cmp_defaultcmp_short_succ() {
        assert_eq!(
            DefaultCmp.find_short_succ("abcd".as_bytes()),
            "b".as_bytes()
        );
        assert_eq!(
            DefaultCmp.find_short_succ("zzzz".as_bytes()),
            "{".as_bytes()
        );
        assert_eq!(DefaultCmp.find_short_succ(&[]), &[0xff]);
        assert_eq!(
            DefaultCmp.find_short_succ(&[0xff, 0xff, 0xff]),
            &[0xff, 0xff, 0xff, 0xff]
        );
    }

    #[test]
    fn test_cmp_separator_stays_between_the_keys() {
        use crate::key_types::ValueType;
        let cmp = InternalKeyCmp(Rc::new(Box::new(DefaultCmp)));
        let key =
            |user: &[u8], seq, typ| LookupKey::new_full(user, seq, typ).internal_key().to_vec();
        // Nothing sorts between a key and the same key with a zero byte appended, between
        // two entries of one key, or above the empty key and below "\0": the block's last
        // key is the separator then, whatever the sequence numbers and types.
        let pairs: [(&[u8], &[u8]); 4] = [
            (b"abc", b"abc\0"),
            (b"abc", b"abc"),
            (b"", b"\0"),
            (b"\xff\xff", b"\xff\xff\0"),
        ];
        for (a, b) in pairs {
            for (seq_a, seq_b) in [(1, 2), (9, 2)] {
                if a == b && seq_a < seq_b {
                    // One key's entries are stored newest first.
                    continue;
                }
                for typ in [ValueType::TypeValue, ValueType::TypeDeletion] {
                    let (a, b) = (key(a, seq_a, typ), key(b, seq_b, ValueType::TypeValue));
                    let sep = cmp.find_shortest_sep(&a, &b);
                    assert_eq!(sep, a);
                    assert!(cmp.cmp(&sep, &b) == Ordering::Less);
                }
            }
        }
        assert_eq!(DefaultCmp.find_shortest_sep(b"abc", b"abc\0"), b"abc");
        assert_eq!(DefaultCmp.find_shortest_sep(b"", b"\0"), b"");
        assert_eq!(DefaultCmp.find_shortest_sep(b"", b"x"), b"");
    }

    #[test]
    fn test_cmp_internalkeycmp_shortest_sep() {
        let cmp = InternalKeyCmp(Rc::new(Box::new(DefaultCmp)));
        assert_eq!(
            cmp.find_shortest_sep(
                LookupKey::new("abcd".as_bytes(), 1).internal_key(),
                LookupKey::new("abcf".as_bytes(), 2).internal_key()
            ),
            LookupKey::new("abce".as_bytes(), 1).internal_key()
        );
        assert_eq!(
            cmp.find_shortest_sep(
                LookupKey::new("abcd".as_bytes(), 1).internal_key(),
                LookupKey::new("abce".as_bytes(), 2).internal_key()
            ),
            LookupKey::new("abcd\0".as_bytes(), 1).internal_key()
        );
        assert_eq!(
            cmp.find_shortest_sep(
                LookupKey::new("abc".as_bytes(), 1).internal_key(),
                LookupKey::new("zzz".as_bytes(), 2).internal_key()
            ),
            LookupKey::new("b".as_bytes(), types::MAX_SEQUENCE_NUMBER).internal_key()
        );
        assert_eq!(
            cmp.find_shortest_sep(
                LookupKey::new("abc".as_bytes(), 1).internal_key(),
                LookupKey::new("acd".as_bytes(), 2).internal_key()
            ),
            LookupKey::new("abd".as_bytes(), 1).internal_key()
        );
        assert_eq!(
            cmp.find_shortest_sep(
                LookupKey::new("abc".as_bytes(), 1).internal_key(),
                LookupKey::new("abe".as_bytes(), 2).internal_key()
            ),
            LookupKey::new("abd".as_bytes(), 1).internal_key()
        );
        assert_eq!(
            cmp.find_shortest_sep(
                LookupKey::new("".as_bytes(), 1).internal_key(),
                LookupKey::new("".as_bytes(), 2).internal_key()
            ),
            LookupKey::new("".as_bytes(), 1).internal_key()
        );
        assert_eq!(
            cmp.find_shortest_sep(
                LookupKey::new("abc".as_bytes(), 2).internal_key(),
                LookupKey::new("abc".as_bytes(), 2).internal_key()
            ),
            LookupKey::new("abc".as_bytes(), 2).internal_key()
        );
    }

    #[test]
    fn test_cmp_internalkeycmp() {
        let cmp = InternalKeyCmp(Rc::new(Box::new(DefaultCmp)));
        // a < b < c
        let a = LookupKey::new("abc".as_bytes(), 2).internal_key().to_vec();
        let b = LookupKey::new("abc".as_bytes(), 1).internal_key().to_vec();
        let c = LookupKey::new("abd".as_bytes(), 3).internal_key().to_vec();
        let d = "xyy".as_bytes();
        let e = "xyz".as_bytes();

        assert_eq!(Ordering::Less, cmp.cmp(&a, &b));
        assert_eq!(Ordering::Equal, cmp.cmp(&a, &a));
        assert_eq!(Ordering::Greater, cmp.cmp(&b, &a));
        assert_eq!(Ordering::Less, cmp.cmp(&a, &c));
        assert_eq!(Ordering::Less, cmp.cmp_inner(d, e));
        assert_eq!(Ordering::Greater, cmp.cmp_inner(e, d));
    }

    #[test]
    #[should_panic]
    fn test_cmp_memtablekeycmp_panics() {
        let cmp = MemtableKeyCmp(Rc::new(Box::new(DefaultCmp)));
        cmp.cmp(&[1, 2, 3], &[4, 5, 6]);
    }
}
