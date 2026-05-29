
I think that the int , tez and nat are all serialized the same way. But what is the algorithm? I tried running some examples through the LIGO CLI but couldn't figure out the system $ ligo interpret -s pascaligo 'Bytes.pack(1n)' 0x050001 0001 is 1 in dec. $ ligo interpret -s pascaligo 'Bytes.pack(1000000n)' 0x050080897a 080897a is 8.423.802 in dec.
