
When I call the CONTRACT opcode in Michelson, say I have a command like: CONTRACT int; Does the contract I'm calling this on have to have int as a param or could it be something like: parameter Or(Or(string, bool), Or((pair nat int), int));
