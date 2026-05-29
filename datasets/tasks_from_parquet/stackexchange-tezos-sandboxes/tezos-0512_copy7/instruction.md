
In my smart contract in LIGO I serialize some data using Bytes.pack which calls the underlying PACK Michelson instruction. In my case, the function signature is (nat, address, nat) -> bytes . Does any JavaScript/TypeScript library like Taquito have a functionality to emulate this LIGO instruction or do I have to implement that myself?
