
I'm trying to get tuple value by index using contract's input param. Getting weird assertion error. let%entry main = (idx: int, storage) => { let tmp = ("test - 1","test - 2", 3) failwith(tmp[idx]) Unhandled exception "Assert_failure ./liquidity/tools/liquidity/liquidCheck.ml:1890:7" But accessing via just int works failwith(tmp[1]) . Do we have any limitations here?
