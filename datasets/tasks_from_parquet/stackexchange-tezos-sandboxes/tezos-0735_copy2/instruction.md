
I want to turn a string into the 0x byte format inside an entrypoint. If I try to use sp.pack() the actual value I want gets prepended with pack instructions as detailed here I'm looking for a runtime equivalent of sp.utils.bytes_of_string() (which can only be used at compile time) that does not prepend anything to the string. Is it possible to do this inside an entrypoint with smartpy?
