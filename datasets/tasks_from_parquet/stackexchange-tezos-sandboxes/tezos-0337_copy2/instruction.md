
When I implement a getBalance in the FA1.2 function like this: @sp.entryPoint def getBalance(self, params): return self.data.balances[params.addr] As test: scenario += c1.getBalance(addr = alice).run(sender = alice) I get the above error: Error: Type error, (sp.TRecord(addr = sp.TAddress)) is not (sp.TUnit)
