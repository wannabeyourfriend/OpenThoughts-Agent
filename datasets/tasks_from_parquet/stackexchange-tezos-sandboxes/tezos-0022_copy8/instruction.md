
Documentation states the following: tez : an unbounded positive float of Tezzies, written either with a tz suffix (1.00tz, etc.) or as a string with type coercion ("1.00" : tez). Yet the following example in Liquidity produces an error: if amount While type coercion for nat works as expected: Github issue can be found here
