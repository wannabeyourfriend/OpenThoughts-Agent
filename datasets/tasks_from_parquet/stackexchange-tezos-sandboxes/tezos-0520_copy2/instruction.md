
Running the PACK operation on a tuple (represented as a pair in Michelson, I get $ ligo interpret -s pascaligo 'Bytes.pack((1, 2))' 0x05070700010002 The numbers 1 and 2 are serialized as 0001 and 0002 , respectively. The tuple (1,2) is represented as PAIR 1 2 in Michelson and PAIR serializes to 0x07 . So why are there two 0x07 values, and not just one?
