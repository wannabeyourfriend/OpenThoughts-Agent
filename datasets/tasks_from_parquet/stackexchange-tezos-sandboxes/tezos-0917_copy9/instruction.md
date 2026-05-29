
I defined an entrypoint with 2 arguments: @sp.entrypoint def update(self, newx, newy): self.data.x = newx self.data.y = newy then I wrote a test of this entrypoint: scenario += contract.update(newx = 4, newy = 3) Do I always have to write the argument names? Why is contract.update(4, 3) not possible? Question from Slack
