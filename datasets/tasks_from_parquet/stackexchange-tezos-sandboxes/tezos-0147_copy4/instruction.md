
My smart contract compiles but when I use any of the entry methods it throws "Unhandled exception (Invalid_argument List.fold_left2)". It seems to have happened when I added the owner parameter into the storage init. type storage = { something: (address, nat list) map; is_on: bool; owner: address; } let%init storage (owner: address) = { something = Map []; is_on = true; owner; }
