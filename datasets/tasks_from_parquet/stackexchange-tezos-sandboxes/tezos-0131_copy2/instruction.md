
I have a contract with storage as let%init storage = (Map : (key_hash, nat)map) I want to deploy the contract using tezos-client originate contract command. I am unable to init storage, tried several options. --init '(Pair {} (Pair {} 0))' I get the following error: Ill typed data: 1: (Pair {} (Pair {} 0)) is not an expression of type map key_hash nat
