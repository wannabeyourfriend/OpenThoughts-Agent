
Does LIGO support parameterized types? Consider rcons , which is like cons ( :: ) but with the arguments flipped. I'd like to be able to write something like: let rcons((xs, x): T list * T): T list = x :: xs where T is an arbitrary type. Is this possible?
