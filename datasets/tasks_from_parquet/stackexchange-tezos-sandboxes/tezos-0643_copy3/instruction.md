
Error: Type sp.TInt / sp.TNat mismatch sp.is_nat expects a sp.TInt Got: sp.TNat line 119 Line: 119 self.data.shop_items[item_to_purchase.key].amount -= sp.as_nat(item_to_purchase.value) I'm not sure where the sp.TInt is coming from... All the numbers are supposed to be sp.TNat
