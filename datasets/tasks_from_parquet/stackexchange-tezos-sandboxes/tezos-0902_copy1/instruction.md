
How to call a getter entrypoint in contract using completium-cli? For example if it is declared like this: variable bar : nat = 0 variable msg : string = "" getter getBar(s : string) : nat { msg := s; return (bar + length(s)) }
