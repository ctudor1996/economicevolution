import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
# import gc

# Parameters
initWorkerPopulation = 1000
marriageRate = 0.5
deathAge = 70
childrenPerHh = 1.5
initNrFirms = 100
seniorAge = 60
	# for living costs
cphi = 10.0 # capital phi
comega = 10 # capital omega
	# firm related
lphi = 0.5 # lowercase phi, relative importance of H,K in production
psi = 0.3 # capital psi, price mark-up adjustment
alpha_f = 0.5 # Investment in R&D (in the document it appears as 1-alpha_f)
theta = 0.8 # theta, proportion of worker productivity for wage
lomega = 1.0 # lowercase omega, scaling with productivity
s = 1.5 # scales with number of workers
o = 3.0 # scales with number of workers
lgamma = 0.5 # lowercase gamma, rate of capital progress

	#Bank Values
lRate = 1.2 # costs of debts
iRate = 1.05 # return on savings
bRate = 1.1 # government bonds rate
eParam = 0.5 # Entrepreneurship adjustment parameter
z = 2.0 # cost of entrepreneurship

# Lists
hhList = []
humanList = []
firmList = []

# Computed values
meanIncome = 0.0
meanFirmK = 0.0
meanHid = 0.0
aSupply = 0.0 # Aggregated supply, for market dominance
aDemand = 0.0 # Aggregated demand, for sales


def chi(hi):
	return 0.5*hi

###################################################################################
#Begin Human                                                                      #
###################################################################################

class Human:
	def __init__(self,hid,age=0,savings=0):
		self.hid = hid
		# self.hc = hc
		self.age = age
		self.married = False
		self.household = None
		self.adult = False
		self.senior = False

		# Consumption variables
		self.lCosts = 0.0 # Living Costs
		self.wage = 0.0
		self.savings = 0.0
		self.income = 0.0 # it's either wage+savings, or wage+savings-living costs
		self.oldGoods = 0.0
		self.newGoods = 0.0
		# firm related variables
		self.employer = None
		self.pQ = 0 # Productivity as worker
		self.employable = False
		hList.append(self)


	def checkDeath(self,hList):
		if self.age == deathAge:
			hh = self.household
			
			# If the person dies, the other person in the household,
			# if there is anyone else, will automatically become unmarried
			# If the person is unmarried, it doesn't change anything
			for x in hh.members:
				x.married = False
			hh.members.remove(self)
			hList.remove(self) # remove person from list of humans
			# TODO: check what to do with savings. Inheritance???

	def checkRetire(self,hList):
		if self.age == seniorAge:
			# self.wage = 0
			self.employer.employees.remove(self)
			self.senior = True

	def passYear(self):
		self.age+=1
		self.wage = 0
		self.income = 0
		self.oldGoods = 0
		self.newGoods = 0
		self.pQ = 0
		self.checkRetire()
		self.checkDeath()

	def livingCost(self):
		return comega * (self.age-2)**2 + cphi

	def checkEmployable(self):
		if self.employable:
			return

		if self.age > 18**np.sqrt(self.hid/meanHid):
			self.employable = True


###################################################################################
#End HUMAN                                                                        #
###################################################################################

###################################################################################
#Begin Household                                                                  #
###################################################################################

class Household:
	def __init__(self,nr,firstMember):
		self.nr = nr
		self.members = [firstMember]
		self.hhsavings = firstMember.savings
		firstMember.household = self
		self.totalIncomes = 0.0
		self.disposableIncome = 0.0
		hhList.append(self)

	def checkDestroyHh(self,hhList):
		if not self.members:
			hhList.remove(self)

	# Should this method be included in the Human class? Right now
	# it seems it would be easier to compute it when iterating
	# through households

	def acquireHC(self):
		children = [x for x in self.members if x.adult==False]
		if not children:
			return
		earners = [x for x in self.members if x.adult==True]

		for c in children:
			c.hid = c.hid * (1 + chi(self.disposableIncome/meanIncome)**c.age)

	def payLivingCosts(self):
		# you can pay living costs using your savings, so don't need to check the wage
		earners = [x for x in self.members if x.adult==True] 
		totalLivingCosts = float(np.sum([x.lCosts for x in self.members]))
		for e in earners:
			e.income = e.wage + e.savings

		self.totalIncomes = float(np.sum([x.income for x in earners]))
		
		# If there household does not have any kind of incomes, presume
		# that the living costs are covered another way, but utility is not computed
		if self.totalIncomes == 0:
			return

		self.disposableIncome = max(self.totalIncomes - totalLivingCosts,0)
		for e in earners:
			e.income = disposableIncome * e.income/totalIncomes

	def optimizeUtility(self):
		earners = [x for x in self.members if x.adult==True] 

		alpha = 0.45
		def func(x):
			return -(x[0]**alpha*x[1]**(alpha-0.1)*x[2]**(1.1-2*alpha))
		def con(c):
			fc = lambda t: t[0]+t[1]+t[2]-c.income
			return {'type':'eq','fun':fc}

		for e in earners:
			if e.employer:
				if e.income/meanIncome < 1.25:
					alpha = 0.45
				else:
					alpha = 1 - meanIncome/e.income
			else:
				alpha = 0.15

			constraint = con(e)

			res = optimize.minimize(func,[e.income/3,e.income/3,e.income/3],constraints = constraint)
			e.oldGoods = res.x[0]
			e.newGoods = res.x[1]
			e.savings = res.x[2]
			bank.cSavings += e.savings

			# Test if the optimizer accidentally used negative funds for utility computation
			if (res.x<0).any:
				print('Negative detected')


###################################################################################
#End Household                                                                    #
###################################################################################

###################################################################################
#Begin Firm                                                                       #
###################################################################################

class Firm:
	def __init__(self,entrepreneur,capital,debt = 0):
		self.employees = [entrepreneur]
		entrepreneur.employer = self
		self.costs = 0.0
		self.debt = debt
		self.capital = capital
		self.firmProductivity = 0
		self.wages = []
		self.price = 1.0
		self.markup = 1.0
		self.rd = 0.0 # R&D expenditure
		self.goodsType = 0
		self.goodsProduced = 0
		self.goodsSold = 0
		self.marketDominance = 0.0
		self.profit = 0.0
		self.savings = 0.0
		firmList.append(self)


	def setGoodsType(self,mK):
		if self.capital < mK:
			self.goodsType = 0
		else:
			self.goodsType = 1
		# oldGoods = 0, newGoods = 1

	def getFirmProductivity(self):
		self.firmProductivity = 0.0

		for e in self.employees:
			e.pQ = e.hid**lphi * self.capital**(1-lphi)
			self.firmProductivity += e.pQ
			e.wage = theta**(e.hid/meanHid)*e.pQ*self.price
		self.wages = [x.wage for x in self.employees]

	def getFirmCosts(self):
		self.costs = o*len(self.employees)**s *self.capital**(self.capital/meanFirmK) + \
		lomega * self.firmProductivity + \
		np.sum(self.wages)

	def setMarkupAndPrice(self): # Remember to use after selling goods
		if self.goodsSold == self.firmProductivity:
			self.markup = self.markup * (1+psi)
		else:
			self.markup = self.markup*(self.goodsSold/self.firmProductivity)
		self.price = (self.costs/self.firmProductivity)(1+self.markup)

	def loan(self):
		moneyNeeded = self.costs - self.savings
		if moneyNeeded > 0:
			self.debt = self.debt + moneyNeeded*lRate
			self.savings = 0
		else:
			self.savings = -moneyNeeded

	def optimizeUtility(self):

		# Get the profit first
		self.profit = self.goodsSold * self.price + self.savings#- self.debt
		if self.profit > self.debt:
			bank.debts -= self.debt
			bank.funds += self.debt
			self.debt = 0
			self.profit = self.profit - self.debt
			self.savings = 0.0
		else:
			self.debt -= self.profit
			bank.debt -= self.profit
			bank.funds += self.profit
			self.profit = 0
			self.rd = 0
			self.savings = 0.0
			return

		# if self.profit <= 0:
		# 	self.debt = -self.profit
		# 	self.rd = 0.0
		# 	self.profit = 0.0
		# 	return
		# bank.debts -= self.debt
		# self.debt = 0.0


		def func(x): # the exponents are inverted because of the definition of alpha_f
			return -(x[0]**(1-alpha_f)*x[1]**alpha_f)
		def con(c):
			fc = lambda t: t[0]+t[1]-self.profit
			return {'type':'eq','fun':fc}

		constraint = con(e)

		res = optimize.minimize(func,[self.profit/2,self.profit/2],constraints = constraint)
		self.savings += res.x[0]
		bank.fSavings += res.x[0]
		self.rd = res.x[1]

	def capitalProgress(self):
		if self.rd>0:
			self.capital = self.capital + gamma*self.rd/self.capital
			self.rd = 0.0

	def getMarketDominance(self):
		self.marketDominance = self.price*self.firmProductivity / aSupply

	def bankrupcy(self):
		for e in self.employees:
			e.wage = 0
			e.employer = None
		# TODO
		# remove debt from bank

		firmList.remove(self)

	def checkBankrupcy(self):
		if np.sum(self.wages) > self.debts:
			self.bankrupcy()

	def hire(self,h):
		self.employees.append(h)
		h.employer = self

		# If employee is still in his childhood household, he now creates his own
		if h.adult == False:
			h.adult = True
			h.household.members.remove(h)

			hh = Household(0,h)
			hhList.append(hh)




	def fire(self,h):
		h.employer = None
		h.wage = 0.0
		self.employees.remove(h)

	def checkHire(self):
		
		# If not all goods are sold, the firm will not hire anyone
		if self.goodsSold != self.firmProductivity:
			return
		
		candidates = [h for h in hList if h.employable and (not h.senior) and (not h.employer)]
		candidates.sort(key=lambda x: x.hid,reverse=True)
		for c in candidates:
			tempEmployees = self.employees + [c]
			tempPQ = list(map(lambda x: x.hid**lphi * self.capital**(1-lphi),tempEmployees))
			tempWages = list(map(lambda x: theta**(x.hid/meanHid)*self.price, tempEmployees))
			tempWages = np.multiply(tempWages,tempPQ)
			
			tempCost = o*len(tempEmployees)**s *self.capital**(self.capital/meanFirmK) + \
						lomega * np.sum(tempPQ) + \
						np.sum(tempWages)

			# We know the last tempEmployee is our candidate
			if tempPQ[-1] * self.price > tempCost - self.costs: 
				self.hire(c)
				break

	def checkFire(self):
		if self.debt > 0 and self.firmProductivity != self.goodsSold:
			e = self.employees.copy()
			e.sort(key=lambda x: x.hid)
			self.fire(e[0])
			

###################################################################################
#End Firm                                                                         #
###################################################################################

###################################################################################
#Begin Bank                                                                       #
###################################################################################

class Bank:
	def __init__(self):
		self.funds = 10000.0
		self.cSavings = 0.0 # consumer savings
		self.fSavings = 0.0 # firm savings
		self.debts = 0.0 # firm debts
		self.bonds = 0.0
		self.balance = self.funds - self.cSavings + self.debts + self.bonds
		self.investment = 0.0

	# When people will pay their living costs and optimize their utility,
	# they will withdraw all their money
	def withdrawSavings(self):
		for h in hList:
			h.savings = iRate*h.savings
		self.funds -= self.cSavings*iRate
		self.cSavings = 0

		for f in firmList:
			f.savings = iRate * f.savings
		self.funds -= self.fSavings*iRate
		self.fSavings = 0

	def getBalance(self):
		self.balance = self.funds - self.cSavings - self.fSavings + self.debts + self.bonds

	def entrepreneurship(self):
		# Get possible entrepreneurs
		entr = [x for x in hList if not x.senior and x.employable and not x.employer]
		entr.sort(key: lambda x: x.hid, reverse=True)
		tCapital = []
		establishingCosts = []
		for e in entr:
			currentTCapital = e.hid**(meanFirmK/meanHid)
			tCapital.append(currentTCapital)
			establishingCosts.append((currentTCapital**z)/meanFirmK)

		# Since the list is descending, and higher Hids are prioritized, then, to simplify
		# things, we just add each possible new firm to the total costs
		currentCosts = 0.0
		for i in length(entr):
			if currentCosts + establishingCosts[i] <= self.investment:
				currentCosts += establishingCosts[i]
				Firm(entr[i],entr[i].hid)
				




###################################################################################
#End Bank                                                                         #
###################################################################################



###################################################################################
#Begin Other Functions                                                            #
###################################################################################



def joinHouseholds(hh1,hh2):
	hh1.members.append(hh2.members[0]) # Second member comes to the first household
	hh2.members[0].household = hh1 # Change the new member's household (i.e., new ID)
	hh1.hhsavings += hh2.members[0].savings

	hh2.members.remove(hh2.members[0]) # Remove from old household
	hh2.checkDestroyHh(hhList) # Destroy the now empty second household

	for x in hh2.members:
		x.married = True



def getMarriedHouseholds(hh):
	marriedHh = [x for x in hh if len(x.members) >=2]
	return marriedHh

def updateMeanIncome():
	e = list(filter(lambda x: x.adult), humanList)
	meanIncome = np.mean([x.income for x in e])

def sales():
	
	# Should be out of the function
	meanFirmK = np.mean([f.capital for f in firmList])
	for f in firmList:
		f.setGoodsType(meanFirmK)
		f.getFirmProductivity()

	###########
	#old goods#
	###########



	# get aggregated demand and supply
	aDemand = np.sum([h.oldGoods for h in hList])
	currentFirms = [f for f in firmList if f.goodsType == 0]
	currentFirms.sort(key=lambda x: x.price)
	aSupply = np.sum([f.firmProductivity*f.price for f in currentFirms])

	currentDemand = aDemand

	# compute market dominance for each eligibile firm
	for f in currentFirms:
		f.getMarketDominance()
		f.goodsSold = 0.0

	while True:
		for f in currentFirms:
			if f.goodsSold <f.firmProductivity:
				# Because of truncations caused by division, there's a change that sold and produced goods may never be equal so
				# create special conditions for finishing the stock
				getSaleCase = np.argmin([(f.firmProductivity-f.goodsSold)*f.price,currentDemand,f.marketDominance*f.firmProductivity*f.price])
				if getSaleCase == 0 : # if there is still demand but the firm will finish selling all its stock
					aSupply -= (f.firmProductivity-f.goodsSold)*f.price
					f.goodsSold = f.firmProductivity
				elif getSaleCase == 1:
					f.goodsSold += aSupply/f.price
					currentDemand = 0
					break
				else:
					aSupply -= f.marketDominance*f.firmProductivity*f.price
					f.goodsSold += f.marketDominance*f.firmProductivity

		if currentDemand == 0:
			break
		if [f.goodsSold == f.firmProductivity for f in currentFirms].all():
			# there is still unsatisfied demand
			# people will get a their money back, based on their proportion of the demand
				for h in hList:
					h.savings += h.oldGoods/aDemand * currentDemand

	###########
	#new goods#
	###########



	# get aggregated demand and supply
	aDemand = np.sum([h.newGoods for h in hList])
	currentFirms = [f for f in firmList if f.goodsType == 1]
	currentFirms.sort(key=lambda x: x.price)
	aSupply = np.sum([f.firmProductivity*f.price for f in currentFirms])

	currentDemand = aDemand

	# compute market dominance for each eligibile firm
	for f in currentFirms:
		f.getMarketDominance()
		f.goodsSold = 0.0

	while True:
		for f in currentFirms:
			if f.goodsSold <f.firmProductivity:
				# Because of truncations caused by division, there's a change that sold and produced goods may never be equal so
				# create special conditions for finishing the stock
				getSaleCase = np.argmin([(f.firmProductivity-f.goodsSold)*f.price,currentDemand,f.marketDominance*f.firmProductivity*f.price])
				if getSaleCase == 0 : # if there is still demand but the firm will finish selling all its stock
					aSupply -= (f.firmProductivity-f.goodsSold)*f.price
					f.goodsSold = f.firmProductivity
				elif getSaleCase == 1:
					f.goodsSold += aSupply/f.price
					currentDemand = 0
					break
				else:
					aSupply -= f.marketDominance*f.firmProductivity*f.price
					f.goodsSold += f.marketDominance*f.firmProductivity

		if currentDemand == 0:
			break
		if [f.goodsSold == f.firmProductivity for f in currentFirms].all():
			# there is still unsatisfied demand
			# people will get a their money back, based on their proportion of the demand
				for h in hList:
					h.savings += h.newGoods/aDemand * currentDemand









###################################################################################
#End Other Functions                                                              #
###################################################################################




###################################################################################
#Initialization                                                                   #
###################################################################################

for i in range(initWorkerPopulation):
	h = Human(np.random.normal(50,5),np.random.randint(25,50))
	h.adult = True
	h.employable = True
	humanList.append(h)
# Create initial households:
for x in humanList:
	hhList.append(Household(0,x))
humanList.sort(key = lambda x: x.age)


# Initial marriages

initMarriages = np.random.permutation(initWorkerPopulation)
initMarriages = initMarriages[:int(initWorkerPopulation*marriageRate)]
initMarriages.sort()

for i in range(0,len(initMarriages),2):
	joinHouseholds(humanList[initMarriages[i]].household,humanList[initMarriages[i+1]].household)


# Initialize children

marriedHh = getMarriedHouseholds(hhList)
initNrChildren = int(np.round(childrenPerHh * len(marriedHh)))
initChildrenList = []

for i in range(initNrChildren):
	initChildrenList.append(Human(np.random.normal(50,5),np.random.randint(0,18)))

initChildrenList.sort(key = lambda x: x.age)
marriedHh.sort(key = lambda x: x.members[0].age)


# Associate children to households, make older children part of households with older members
itHh = 0
for c in initChildrenList:
	
	marriedHh[itHh].members.append(c)
	itHh += 1
	if itHh == len(marriedHh):
		itHh = itHh//2

humanList = humanList + initChildrenList

# Create firms

# Select a number of entrepreneurs
# We don't have any seniors, and everyone is unemployed at this moment, so we just
# need to get adults
entrepreneurs = list(filter(lambda x: x.adult==True and x.senior==False, humanList))
entrepreneurs = np.random.choice(entrepreneurs,initNrFirms,replace = False)
for e in entrepreneurs:
	f = Firm(e,e.hid)
	firmList.append(f)




