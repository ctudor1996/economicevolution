import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from time import time
import pdb
import matplotlib.pyplot as plt
# import gc

# Parameters
initWorkerPopulation = 1000
marriageRate = 0.6
deathAge = 70
childrenPerHh = 1.5
initNrFirms = 100
seniorAge = 60
firmSizeMean = [30,15,5] # Large firms have a mean of 25 workers, medium of 10 and small of 4
firmSizeDeviation = [5,3,1] # Standard deviation of firm size means
initialFirmSizeDistributions = [0.1,0.25,0.5] # Reads as 10% of workforce in firms of size N(25,2), 30% in firms of size N(10,2) and 60% in firms of size N(4,1)
 

# Other human parameters
phiG = 1.25 # constant to adjust no. of children per Hh

	# for living costs
cphi = 10.0 # capital phi
comega = 0.05 # capital omega
chi = 0.005 # adjusts human capital growth


	# firm related
lphi = 0.5 # lowercase phi, relative importance of H,K in production
psi = 0.1 # capital psi, price mark-up adjustment
alpha_f = 0.1    # Investment in R&D (in the document it appears as 1-alpha_f)
theta = 2 #0.8 # theta, proportion of worker productivity for wage


lomega = 0.8 # lowercase omega, scaling with productivity
s = 1.2 # scales with number of workers
o = 0.2 # scales with number of workers
lgamma = 0.35 # lowercase gamma, rate of capital progress
betaF = 0.3 # modifier for fraction of debt to be covered from wages of the fired employees
maxIncrease = 0.2 # Maximum increase in firm size through hiring

	#Bank Values
lRate = 1.03 # costs of debts
iRate = 1.02 # return on savings
bRate = 1.2 # government bonds rate
eParam = 0.03 # Entrepreneurship adjustment parameter
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
nFirms = 0 # New firms in a year
nBankruptFirms = 0 # Bankrupt firms in a year (includes firms left with no workers)

# def chi(hi):
# 	return 0.5*hi

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
		self.goods = 0.0
		# firm related variables
		self.employer = None
		self.pQ = 0 # Productivity as worker
		self.employable = False
		humanList.append(self)


	def checkDeath(self):
		if self.age == deathAge:
			hh = self.household
			
			# If the person dies, the other person in the household,
			# if there is anyone else, will automatically become unmarried
			# If the person is unmarried, it doesn't change anything
			for x in hh.members:
				x.married = False
			hh.members.remove(self)
			humanList.remove(self) # remove person from list of humans
			# TODO: check what to do with savings. Inheritance???

	def checkRetire(self):
		if self.age == seniorAge:
			# self.wage = 0
			if self.employer:
				empl = self.employer
				self.employer.employees.remove(self)
				if len(empl.employees) == 0:
					empl.checkBankrupcy()
			self.senior = True

	def checkEmployable(self):
		if self.employable:
			return

		if self.age > 18*(self.hid/meanHid)**(1/5):
			self.employable = True
			self.adult = True


	def passYear(self):
		self.age+=1
		#print('Wage ',self.wage)
		self.wage = 0
		#self.income = 0
		self.oldGoods = 0
		self.newGoods = 0
		self.pQ = 0
		self.checkRetire()
		self.checkDeath()
		self.livingCosts()
		self.checkEmployable()

	def livingCosts(self):
		self.lCosts = comega * (self.age-30)**2 + cphi




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

	def checkDestroyHh(self):
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
			c.hid = c.hid * (1 + (0.1*(self.totalIncomes/meanHhIncome)**chi)**np.log(c.age+1))
			#print('HID', c.hid)

	def payLivingCosts(self):
		# you can pay living costs using your savings, so don't need to check the wage
		earners = [x for x in self.members if x.adult==True] 
		totalLivingCosts = float(np.sum([x.lCosts for x in self.members]))
		for e in earners:
			if not e.employer:
				e.wage = 0.5*np.min([x.wage for x in humanList if x.employer])
			e.income = e.wage + e.savings

		self.totalIncomes = float(np.sum([x.income for x in earners]))
		
		# If there household does not have any kind of incomes, presume
		# that the living costs are covered another way, but utility is not computed
		if self.totalIncomes == 0:
			return

		self.disposableIncome = max(self.totalIncomes - totalLivingCosts,0)
        
		checkEmployed = np.array([1 if e.employer else 0 for e in earners])
		checkEmployed = checkEmployed.all()
    
		for e in earners:
			if checkEmployed:
				e.income = self.disposableIncome * e.income/self.totalIncomes
			else:    
				e.income = self.totalIncomes/len(earners)

	def optimizeUtility(self):
		earners = [x for x in self.members if x.adult==True] 

		alpha = 0.45
		# beta = 0.8alpha, and since alpha+beta+gamma = 1 => gamma = 1-1.8alpha
		# => alpha = (1-gamma)/1.8
		def func(x):
			return -(x[0]**(alpha)*x[1]**(1-alpha))
			print(alpha)
		def con(c):
			fc = lambda t: t[0]+t[1]-c.income
			fineq1 = lambda t: t[0]
			fineq2 = lambda t: t[1]
			return [{'type':'eq','fun':fc},
			{'type':'ineq','fun':fineq1},
			{'type':'ineq','fun':fineq2}]

		# for e in earners:
		# 	if e.employer:
		# 		if e.income/meanIncome < 2:
		# 			alpha = 0.5
		# 		elif e.income/meanIncome>=2 and e.income/meanIncome<=4:
		# 			alpha = 1/(e.income/meanIncome)
		# 		else:
		# 			alpha = 0.25
		# 	else:
		# 		alpha = 0.4

		for e in earners:
			if e.employer:
				# 
				gamma = np.min([0.1 + (self.disposableIncome/getMeanDisposableIncome())*0.1, 0.8])
				alpha = 1-gamma
			else:
				gamma = np.min([0.3 + (self.disposableIncome/getMeanDisposableIncome())*0.1, 0.8])
				alpha = 1-gamma
			#print('GAMMA',gamma,'disp inc',self.disposableIncome,'mean disp income ', getMeanDisposableIncome(),'alternate ',0.15 + (self.disposableIncome/getMeanDisposableIncome())*0.1)
			# def func(x):
			# 	return -(x[0]**alpha*x[1]**(alpha-0.1)*x[2]**(1.1-2*alpha))
			# def con(c):
			# 	fc = lambda t: t[0]+t[1]+t[2]-c.income
			# 	return {'type':'eq','fun':fc}
			constraint = con(e)
			for i in range(1000):
				w1,w2 = np.random.randint(1,15,2)
				wSum = w1+w2
				w1 = float(w1)/wSum
				w2 = float(w2)/wSum
	
				res = optimize.minimize(func,[w1*e.income,w2*e.income],constraints = constraint)

				if res.x[0] == res.x[0] and res.x[1] == res.x[1]:
					if res.x[0]>=0 and res.x[1]>=0:
						break
				if i==999:
					print(e.income)
					print('COULDN\'T OPTIMIZE')
					exit()

			# print(e.income,res.x[0],res.x[1],res.x[2])
			e.goods = res.x[0]
			e.savings = res.x[1]
			bank.cSavings += e.savings

			# Test if the optimizer accidentally used negative funds for utility computation
			if (res.x<0).any():
				print('Negative detected')
			if (res.x!=res.x).any():
				print('Nan detected')


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
		self.goodsSold = 0.0
		self.marketDominance = 0.0
		self.profit = 0.0
		self.savings = 0.0
		self.aSupply = 0.0 #This two shouldn't be in the class, but it makes things easier
		self.aDemand = 0.0
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
			e.pQ = 6*(e.hid**lphi * self.capital**(1-lphi))/meanFirmK
			self.firmProductivity += e.pQ
		
			
			
			e.wage = (e.hid*self.capital/ meanHid) * theta**(1-getUnemploymentRate()) ## where u is the unemployment rate
			#print('W ',e.wage)		
		if self.firmProductivity <0:
			print('Productivity < 0')
		self.wages = [x.wage for x in self.employees]

		#print('productivity',self.firmProductivity)

	def getFirmCosts(self):
		self.costs = 0.01*(self.firmProductivity * self.capital ** (self.capital/(4*meanFirmK)) * (2.2*np.sin(0.032*self.firmProductivity) + 4)) + np.sum(self.wages)
        
		#print ('fraction of wages in costs',np.sum(self.wages)/self.costs)

	def setMarkupAndPrice(self): # Remember to use after selling goods
		if self.firmProductivity == 0:
			self.checkBankrupcy()
			return
		if self.goodsSold == self.firmProductivity:
			self.markup = self.markup * (1+psi)
		else:
			self.markup = self.markup*(self.goodsSold/self.firmProductivity)
            
		if self.goodsSold == 0: # Initial markup for new firms. Automatically resolves itself after the first year.
			self.markup = 1.1
		self.price = (self.costs/self.firmProductivity)*(1+self.markup)
		if self.price < 0:
			print('Negative price',self.costs,self.firmProductivity)
		#print('markup',self.markup)
		# IF NOTHING IS SOLD, THE MARKUP IS ZERO

	def loan(self):
		moneyNeeded = self.costs - self.savings
		if moneyNeeded > 0:
			self.debt = self.debt + moneyNeeded*lRate
			self.savings = 0
		else:
			self.savings = -moneyNeeded


	def optimizeUtility(self):

		# Get the profit first
		self.profit = self.goodsSold * self.price + self.savings #- self.costs
		if self.profit > self.debt:
			bank.debts -= self.debt
			bank.repaidDebts += self.debt
			bank.funds += self.debt
			self.debt = 0
			self.profit = self.profit - self.debt
			self.savings = 0.0
		else:
			self.debt -= self.profit
			bank.debts -= self.profit
			bank.funds += self.profit
			bank.repaidDebts += self.profit
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
		def con():
			fc = lambda t: t[0]+t[1]-self.profit
			fineq1 = lambda t: t[0]
			fineq2 = lambda t: t[1]
			return [{'type':'eq','fun':fc},
           {'type':'ineq','fun':fineq1},
           {'type':'ineq','fun':fineq2}]

		constraint = con()
		for i in range(100):
			w1,w2 = np.random.randint(1,15,2)
			wSum = w1+w2
			w1 = float(w1)/wSum
			w2 = float(w2)/wSum

			res = optimize.minimize(func,[w1*self.profit,w2*self.profit],constraints = constraint)
			if res.x[0] == res.x[0] and res.x[1] == res.x[1]:
				if res.x[0]>=0 and res.x[1]>=0:
					break

			if i == 99:
				print('COULDN\'T OPTIMIZE FIRM')
				exit()

		self.savings += res.x[0]
		bank.fSavings += res.x[0]
		self.rd = res.x[1]

	def capitalProgress(self):
		if self.rd>0:
			# print('Old capital ',self.capital)
			self.capital = self.capital + np.min([lgamma*self.rd/self.capital,1.0])
			# print('New capital ',self.capital)
			self.rd = 0.0

	def getMarketDominance(self):
		self.marketDominance = self.price*self.firmProductivity / self.aSupply

	def bankrupcy(self):
		for e in self.employees:
			e.wage = 0
			e.employer = None
		global nBankruptFirms
		nBankruptFirms += 1
		#print(nBankruptFirms)
		# TODO
		# remove debt from bank

		firmList.remove(self)

	def checkBankrupcy(self):
		if len(self.employees)==0:
			self.bankrupcy()
			#print('bankrupt')
			return
		#print('wages',self.wages)
		#print('debt',self.debt)
		if 2*np.sum(self.wages) < self.debt:
			self.bankrupcy()
			#print('bankrupt money')
			

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
		
		candidates = [h for h in humanList if h.employable and (not h.senior) and (not h.employer)]

		if not candidates:
			return

		candidates.sort(key=lambda x: x.hid,reverse=True)
		tempScores = []
		for c in candidates:
			tempEmployees = self.employees + [c]
			tempPQ = list(map(lambda x: x.hid**lphi * self.capital**(1-lphi),tempEmployees))
			tempWages = list(map(lambda x: theta**(x.hid/meanHid)*self.price, tempEmployees))
			#tempWages = np.multiply(tempWages,tempPQ)

			
			tempCost = 0.1*(np.sum(tempPQ) * self.capital ** (self.capital/(4*meanFirmK)) * (2*np.sin(0.04*np.sum(tempPQ)) + 5)) + np.sum(tempWages)
			

			# We know the last tempEmployee is our candidate
			tempScores.append(tempPQ[-1] * self.price - (tempCost - self.costs))

		scoreIndices = np.argsort(tempScores)
		scoreIndices = scoreIndices[::-1]
		maxNrOfNewWorkers = int(np.ceil(maxIncrease*len(self.employees)))
		

		adults = len([x for x in humanList if x.adult and (not x.senior)])
		peopleEmployed = len([x for x in humanList if x.adult and (not x.senior) and x.employer])
		peopleTillFivePercentUnemployment = int(0.95*adults - peopleEmployed)
		if peopleTillFivePercentUnemployment < 1 :
			return

		for i in range(np.min([maxNrOfNewWorkers,len(candidates),peopleTillFivePercentUnemployment])):

			try:
				currentCandidatePos = scoreIndices[i]
			except:
				return

			if tempScores[currentCandidatePos] > 0:
				self.hire(candidates[currentCandidatePos])
			else:
				break


		# # # OLD STYLE HIRING
		# bestCandidate = np.argmax(tempScores)
		# #print('tempscores',tempScores)
		# if tempScores[bestCandidate] > 0:
		# 	self.hire(candidates[bestCandidate])
		# # 	# print('HIRED')

	def checkFire(self):
		if self.debt > 0 and self.firmProductivity != self.goodsSold:
			e = self.employees.copy()
			e.sort(key=lambda x: x.hid)
			wageFired = e[0].wage
			self.fire(e[0])
			e.remove(e[0])

			if not e:
				return

			while wageFired < betaF * self.debt:
				wageFired += e[0].wage
				self.fire(e[0])
				e.remove(e[0])

				if not e:
					# self.checkBankrupcy()
					return

			#print('You\'re fired')
			

###################################################################################
#End Firm                                                                         #
###################################################################################

###################################################################################
#Begin Bank                                                                       #
###################################################################################

class Bank:
	def __init__(self):
		self.funds = 100000.0
		self.cSavings = 0.0 # consumer savings
		self.fSavings = 0.0 # firm savings
		self.debts = 0.0 # firm debts
		self.repaidDebts = 0.0 # money firms repaid this year
		self.bonds = 0.0
		self.balance = self.funds - self.cSavings + self.debts + self.bonds
		self.investment = 0.0

	# When people will pay their living costs and optimize their utility,
	# they will withdraw all their money
	def withdrawSavings(self):
		for h in humanList:
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
		entr = [x for x in humanList if not x.senior and x.employable and not x.employer]
		entr.sort(key= lambda x: x.hid, reverse=True)
		tCapital = []
		establishingCosts = []
		for e in entr:
			currentTCapital = e.hid**(meanFirmK/meanHid)
			tCapital.append(currentTCapital)
			establishingCosts.append((currentTCapital**z)/meanFirmK)

		# Since the list is descending, and higher Hids are prioritized, then, to simplify
		# things, we just add each possible new firm to the total costs
		currentCosts = 0.0
		for i in range(len(entr)):
			if currentCosts + establishingCosts[i] <= self.investment:
				currentCosts += establishingCosts[i]
				Firm(entr[i],entr[i].hid)
				global nFirms
				nFirms += 1

	def bankInvestment(self):

		investmentMoney = self.repaidDebts + self.fSavings + self.cSavings
		# Government bonds
		self.funds += (1-eParam)*investmentMoney*bRate
		# Get investment money for entrepreneurship
		self.investment = eParam * investmentMoney




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
	hh2.checkDestroyHh() # Destroy the now empty second household

	for x in hh1.members:
		x.married = True



def getMarriedHouseholds(hh):
	marriedHh = [x for x in hh if len(x.members) >=2]
	return marriedHh

def updateMeanIncome():
	e = list(filter(lambda x: x.adult, humanList))
	#print('mean income',np.mean([x.income for x in e]))
	return np.mean([x.income for x in e])

def updateMeanHhIncome():
    return np.mean([x.totalIncomes for x in hhList])

def updateMeanFirmK():
	firmK = [f.capital for f in firmList]
	return np.mean(firmK)

def updateMeanHid():
	e = list(filter(lambda x: x.adult, humanList))
	hids = [x.hid for x in e]
	return np.mean(hids)



def sales():
	
	# Should be out of the function
	meanFirmK = np.mean([f.capital for f in firmList])
	for f in firmList:
		#f.setGoodsType(meanFirmK)
		f.getFirmProductivity()
		#print(f.firmProductivity)

	###########
	#old goods#
	###########


	# get aggregated demand and supply
	aDemand = np.sum([h.goods for h in humanList])
	currentFirms = [f for f in firmList]
	currentFirms.sort(key=lambda x: x.price)
	aSupply = np.sum([f.firmProductivity*f.price for f in currentFirms])

	currentDemand = aDemand

	print('Goods\nAggregated Demand:{}\nAggregated Supply:{}'.format(aDemand,aSupply))
	# compute market dominance for each eligibile firm
	for f in currentFirms:
		#print('firma:',f,'\nangajati: ',len(f.employees))
		f.aSupply = aSupply
		f.aDemand = aDemand
		f.getMarketDominance()
		f.goodsSold = 0.0

	while True:
		for f in currentFirms:

			if f.goodsSold <f.firmProductivity:
				# Because of truncations caused by division, there's a change that sold and produced goods may never be equal so
				# create special conditions for finishing the stock
				getSaleCase = np.argmin([(f.firmProductivity-f.goodsSold)*f.price,currentDemand,f.marketDominance*f.firmProductivity*f.price])
				if getSaleCase == 0 : # if there is still demand but the firm will finish selling all its stock
					currentDemand -= (f.firmProductivity-f.goodsSold)*f.price
					f.goodsSold = f.firmProductivity

				elif getSaleCase == 1:
					f.goodsSold += currentDemand/f.price
					currentDemand = 0

					break
				else:
					currentDemand -= f.marketDominance*f.firmProductivity*f.price
					f.goodsSold += f.marketDominance*f.firmProductivity


		if currentDemand == 0:
			break
		if np.array([f.goodsSold == f.firmProductivity for f in currentFirms]).all():
			# there is still unsatisfied demand
			# people will get a their money back, based on their proportion of the demand
			for h in humanList:
				h.savings += h.goods/aDemand * currentDemand
			break







def getUnemployed():
	u = list(filter(lambda x:x.employable and not x.employer and not x.senior, humanList))
	u.sort(key=lambda x: x.hid, reverse=True)
	return u


def upholdMarriageRate():

	# Get all adults
	allAdults = [x for x in humanList if x.adult == True]
	# Get all married
	allMarried = [x for x in humanList if x.married == True]
	print('adults', len(allAdults),'married',len(allMarried))
	# Check rate
	if float(len(allMarried))/len(allAdults) < 0.5:
		noPeopleToMarry = round(float(len(allAdults))/2) - len(allMarried)
		# Make sure it's an even number
		noPeopleToMarry = noPeopleToMarry + noPeopleToMarry%2

		# All the single people. Extra check to not already have kids in their care
		allSingle = [x for x in humanList if x.married == False and x.adult == True and len(x.household.members)==1]
		# Choose the new couples 
		willMarry = np.random.choice(allSingle, noPeopleToMarry, replace=False)
		# Marry them
		for i in range(0,len(willMarry),2):
			joinHouseholds(willMarry[i].household,willMarry[i+1].household)

def updateChildrenPerHh():
    return phiG + meanHid/meanFirmK
    
def upholdNatalityRate():
	
	# Get number of children (not adults and not employable)
	children = [x for x in humanList if x.adult==False and x.employable==False]
	
	# Children are born into married households
	marriedHh = getMarriedHouseholds(hhList)
	# Younger couples have higher chances of having children
	marriedHh.sort(key = lambda x: x.members[0].age)
	childrenPerHh = phiG + meanHid / meanFirmK
	noHh = len(marriedHh)

	if float(len(children)) / len(marriedHh) < childrenPerHh:
		childrenNeeded = int(noHh * childrenPerHh) - len(children)

		for i in range(min(childrenNeeded,noHh)):
			c = Human(np.random.normal(meanHid,0.1*meanHid))
			marriedHh[i].members.append(c)
			c.household = marriedHh[i]
			# Avoid having multiple children in same year
			
def getUnemploymentRate():
	return float(len([x for x in humanList if x.adult==True and x.senior == False and (not x.employer)])) \
	/len([x for x in humanList if x.adult==True and x.senior == False])

def getMeanDisposableIncome():
	return np.mean([x.disposableIncome for x in hhList])
###################################################################################
#End Other Functions                                                              #
###################################################################################




###################################################################################
#Initialization                                                                   #
###################################################################################

for i in range(initWorkerPopulation):
	h = Human(np.random.normal(50,5),np.random.randint(19,59))
	h.adult = True
	h.employable = True
	h.savings = np.random.normal(100, 10)
# Create initial households:
for x in humanList:
	Household(0,x)
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
	initChildrenList.append(Human(np.random.normal(50,5),np.random.randint(0,17)))

initChildrenList.sort(key = lambda x: x.age)
marriedHh.sort(key = lambda x: x.members[0].age)


# Associate children to households, make older children part of households with older members
itHh = 0
for c in initChildrenList:
	
	marriedHh[itHh].members.append(c)
	c.household = marriedHh[itHh]
	itHh += 1
	if itHh == len(marriedHh):
		itHh = itHh//2

# humanList = humanList + initChildrenList

# Create firms

# Select a number of entrepreneurs
# We don't have any seniors, and everyone is unemployed at this moment, so we just
# need to get adults
possibleEntrepreneurs = list(filter(lambda x: x.adult==True and x.senior==False, humanList))
# Create firms and hire workers based on the distribution of firm sizes
for i,dist in enumerate(initialFirmSizeDistributions):
	# First compute the number of firms given the distributions
	populationCovered = int(dist * initWorkerPopulation)
	currentFirmSize = firmSizeMean[i]
	nrFirms = int(populationCovered / currentFirmSize)

	for j in range(nrFirms):
		if len(possibleEntrepreneurs) == 0: # can't create more firms
			break

		noWorkers = int(np.random.normal(firmSizeMean[i],firmSizeDeviation[i]) - 1) # The first worker will be the entrepreneur
		# Create the actual firm

		f = Firm(possibleEntrepreneurs[0], possibleEntrepreneurs[0].hid)
		possibleEntrepreneurs = possibleEntrepreneurs[1:] # We eliminate the first person from the list
		
		# Hire the rest of the workers
		for k in range(noWorkers):

			if len(possibleEntrepreneurs) == 0: # Workers are taken from the pool of possible entrepreneurs, so if it's empty, the firm has nobody to hire
				break
			f.hire(possibleEntrepreneurs[0])
			possibleEntrepreneurs = possibleEntrepreneurs[1:]

print([len(f.employees) for f in firmList])


# for e in entrepreneurs:
# 	f = Firm(e,e.hid)


# # unemployed = list(filter(lambda x: x.adult==True and x.senior==False and not x.employer, humanList))
# # unemployed.sort(key=lambda x: x.hid, reverse=True)
# unemployed = getUnemployed()


# firmList.sort(key=lambda x: x.capital, reverse=True)
# fList = firmList.copy()

# ## OLD INITIAL HIRINGS
# # for h in unemployed:
# # 	if fList:
# # 		fList[0].hire(h)
# # 		if len(fList[0].employees)==90:
# # 			fList.remove(fList[0])
# # 	else:
# # 		break

# ## NEW INITIAL HIRINGS
# for i, h in enumerate(unemployed):
# 	if i>= 0.9*len(unemployed):
# 		break
# 	# For extra randomness: always shuffle firms and draw the winning firm from a gaussian distribution
# 	# with the mean = to initNrFirms/2 and a standard deviation of 7
# 	fList = np.random.permutation(fList)
# 	winningFirm = np.random.normal(initNrFirms/2,7)
# 	winningFirm = np.min([np.max([0,int(winningFirm)]),initNrFirms-1])
# 	fList[winningFirm].hire(h)

bank = Bank()


GDP = []
histMeanFirmK = []
histHid = []
unemployed = []
fertilityRate = []
histADemandOld = []
histADemandNew = []
newFirms = []
bankruptFirms = []
meanSavings = []
meanFirmSavings = []
meanFirmDebts = []

nrFirms = []
meanWage = []
meanWorkerProductivity = []
meanFirmProductivity = []
meanPrice = []

GDPGrowth = []
priceGrowth = []
wageGrowth = []
GDPByAD = []
###################################################################################
#First year                                                                       #
###################################################################################
# It's different from the normal loop, since there are no hirings, firings, etc.

# print('INITIAL UNEMPLOYMENT ',getUnemploymentRate())
#1. Workers advance one period of life

nFirms = 0
nBankruptFirms = 0
meanHid = updateMeanHid()

t1 = time()
for h in humanList:
	h.passYear()
print('passYear ',time()-t1)
# print('AFTER PASS YEAR UNEMPLOYMENT ',getUnemploymentRate())
#2. Marriage rate and natality rate checked and upheld
#3. Savings accrue at the savings rate
#4. Workers get hired or fired


# 5. Firms take credit to finance production costs
# 6. Production occurs and wages are paid out
meanFirmK = updateMeanFirmK()
meanHid = updateMeanHid()

t1 = time()
for f in firmList:
	f.getFirmProductivity()
	f.getFirmCosts()
	f.setMarkupAndPrice()
	f.loan()


t1=time()
for h in hhList:
	h.payLivingCosts()
print('Pay living costs', time()-t1)

# 8. Increase in human capital occurs based on household income
# 9. Earners receive back proportional remaining household income to maximize their utility between consumption and savings

meanIncome = updateMeanIncome()
meanHhIncome = updateMeanHhIncome()
    
for h in hhList:
	h.acquireHC()

for h in hhList:
	h.optimizeUtility()


# 10. Sales process occurs and goods are bough/sold
sales()
fList = firmList.copy()
fList = list(filter(lambda x: x.goodsType==0,fList))
fList.sort(key=lambda x: x.firmProductivity,reverse=True)
# for f in fList:
# 	print(f)
# 	print(f.costs)
# 	print(f.goodsSold)
# 	print(f.firmProductivity)
# 11. Firms use the revenue earned to pay back debt. Excess money is split between profits (savings for next period) and investment in R&D
# 12. Firms advance Technologically
for f in firmList:
	f.optimizeUtility()
	f.capitalProgress()
	f.checkFire()
	f.checkBankrupcy()

fList = firmList.copy()
np.random.shuffle(fList)
for f in fList:
	f.checkHire()



print(len(hhList),len(humanList),len(firmList))
# 13. The bank calculates amount of money needed to pay back next period based on current savings. Excess profits are used to invest in new firms
bank.withdrawSavings()
bank.bankInvestment()
# 14. New firms are created
# print(len([x for x in humanList if x.adult and not x.employer]))
bank.entrepreneurship()

# print(len(firmList))
# print(len([x for x in humanList if x.adult and not x.employer]))

print('Mean Hid: ', meanHid)
print('Mean income: ', meanIncome)
print('Mean Kapital: ', meanFirmK)
print('No. married households', len(getMarriedHouseholds(hhList)))
print('No. children: ', len([x for x in humanList if x.adult==False]))
print('No. employees per firm: ', [len(x.employees) for x in firmList],'\n\n')
print('Firm revenue: ',[round(x.goodsSold*x.price,2) for x in firmList],'\n\n')
print('Firm income: ',[round(x.profit,2) for x in firmList],'\n\n')
print('Firm debt: ',[round(x.debt,2) for x in firmList],'\n\n')
print('Firm prices: ',[round(x.price,2) for x in firmList],'\n\n')
print('Firm savings: ',[round(x.savings,2) for x in firmList], '\n\n')
print('Firm wages: ',[x.wages for x in firmList],'\n\n')


GDP.append(np.sum([x.price*x.goodsSold for x in firmList]))
histMeanFirmK.append(np.mean([x.capital for x in firmList]))
histHid.append(np.mean([x.hid for x in humanList]))
unemployed.append(getUnemploymentRate())
fertilityRate.append(updateChildrenPerHh())
histADemandOld.append(np.sum([h.oldGoods for h in humanList]))
histADemandNew.append(np.sum([h.newGoods for h in humanList]))
newFirms.append(nFirms)
bankruptFirms.append(nBankruptFirms)
meanSavings.append(np.mean([x.savings for x in humanList if x.adult]))
meanFirmSavings.append(np.mean([x.savings for x in firmList]))
meanFirmDebts.append(np.mean([x.debt for x in firmList]))
nrFirms.append(len(firmList))
meanWage.append(np.mean([x.wage for x in humanList if x.employer]))
meanWorkerProductivity.append(np.mean([x.pQ for x in humanList if x.employer]))
meanFirmProductivity.append(np.mean([x.firmProductivity for x in firmList]))
meanPrice.append(np.mean([x.price for x in firmList]))

###################################################################################
#Main loop                                                                        #
###################################################################################

for i in range(800):

	nFirms = 0
	nBankruptFirms = 0

	# GENERATE INITIAL REPORT
	print('Year: ',i)
	print('No. households: ',len(hhList))
	print('No. persons: ',len(humanList))
	print('No. firms: ',len(firmList))

	#1. Workers advance one period of life
	for h in humanList:
		h.passYear()
	for hh in hhList:
		hh.checkDestroyHh()

	try:
		upholdMarriageRate()
	except:
		pass


	childrenPerHh = updateChildrenPerHh()
	try:
		upholdNatalityRate()
	except:
		pass

	
	#2. Marriage rate and natality rate checked and upheld
	#3. Savings accrue at the savings rate
	#4. Workers get hired or fired


	# 5. Firms take credit to finance production costs
	# 6. Production occurs and wages are paid out
	meanFirmK = updateMeanFirmK()
	meanHid = updateMeanHid()
	for f in firmList:
		f.getFirmProductivity()
		f.getFirmCosts()
		f.setMarkupAndPrice()
		#f.setInitialPrice()
		f.loan()

	# 7. Households use their income to pay living costs 
	#print(np.max([e.wage for e in humanList ]))
	for h in hhList:
		h.payLivingCosts()

	# 8. Increase in human capital occurs based on household income
	# 9. Earners receive back proportional remaining household income to maximize their utility between consumption and savings

	# Test if increase in hid works
	meanIncome = updateMeanIncome()
	meanHhIncome = updateMeanHhIncome()
	#print(meanIncome)
	for h in hhList:
		h.acquireHC()

	for h in hhList:
		h.optimizeUtility()
	fList = firmList.copy()
	fList = list(filter(lambda x: x.goodsType==0,fList))
	fList.sort(key=lambda x: x.firmProductivity,reverse=True)


	# 10. Sales process occurs and goods are bough/sold
	aggregateDemand = np.sum([h.goods for h in humanList])
	sales()

	# 11. Firms use the revenue earned to pay back debt. Excess money is split between profits (savings for next period) and investment in R&D
	# 12. Firms advance Technologically
	for f in firmList:
		#print('Price ',f.price,'Mean wage ',np.mean([x.wage for x in f.employees]))
		f.optimizeUtility()
		f.capitalProgress()
		f.checkFire()
		f.checkBankrupcy()



	fList = firmList.copy()
	np.random.shuffle(fList)
	for f in fList:
		f.checkHire()

		

	# 13. The bank calculates amount of money needed to pay back next period based on current savings. Excess profits are used to invest in new firms
	bank.withdrawSavings()
	bank.bankInvestment()
	# 14. New firms are created
	bank.entrepreneurship()


	# GENERATE FINAL REPORT

	print('Mean Hid: ', meanHid)
	print('Mean income: ', meanIncome)
	print('Mean Kapital: ', meanFirmK)
	print('No. married households', len(getMarriedHouseholds(hhList)))
	print('No. children: ', len([x for x in humanList if x.adult==False]))
	print('No. employees per firm: ', [len(x.employees) for x in firmList],'\n\n')
	print('Firm revenue: ',[round(x.goodsSold*x.price,2) for x in firmList],'\n\n')
	print('Firm income: ',[round(x.profit,2) for x in firmList],'\n\n')
	print('Firm debt: ',[round(x.debt,2) for x in firmList],'\n\n')
	print('Firm prices: ',[round(x.price,2) for x in firmList],'\n\n')
	print('Firm savings: ',[round(x.savings,2) for x in firmList], '\n\n')
	print('Firm wages: ',[x.wages for x in firmList],'\n\n')
	#print('Human income: ', [round(x.income,2) for x in humanList if x.adult==True])
	
	#input('PRESS ANY KEY TO CONTINUE')

	#print('\n\n\n\n')

	GDP.append(np.log(np.sum([x.price*x.goodsSold for x in firmList])))
	histMeanFirmK.append(np.mean([x.capital for x in firmList]))
	histHid.append(np.mean([x.hid for x in humanList])) 
	unemployed.append(getUnemploymentRate())
	fertilityRate.append(updateChildrenPerHh())
	histADemandOld.append(np.sum([h.goods for h in humanList]))
	#histADemandNew.append(np.sum([h.newGoods for h in humanList]))
	newFirms.append(nFirms)
	bankruptFirms.append(nBankruptFirms)
	meanSavings.append(np.log(np.mean([x.savings for x in humanList if x.adult])))
	meanFirmSavings.append(np.log(np.mean([x.savings for x in firmList])))
	meanFirmDebts.append(np.log(np.mean([x.debt for x in firmList])))
	nrFirms.append(len(firmList))
	meanWage.append(np.mean([x.wage for x in humanList if x.employer]))
	meanWorkerProductivity.append(np.mean([x.pQ for x in humanList if x.employer]))
	meanFirmProductivity.append(np.mean([x.firmProductivity for x in firmList]))
	meanPrice.append(np.mean([x.price for x in firmList]))

	GDPGrowth.append((GDP[-1]-GDP[-2])/GDP[-2])
	priceGrowth.append((meanPrice[-1]-meanPrice[-2])/meanPrice[-2])
	wageGrowth.append((meanWage[-1]-meanWage[-2])/meanWage[-2])
	GDPByAD.append(aggregateDemand)


	
plt.figure(figsize=(10,10))
plt.subplot(821)
plt.title('Log GDP')
plt.plot(GDP[200:])

plt.subplot(822)
plt.title('Average Firm Capital')
plt.plot(histMeanFirmK[200:])

plt.subplot(823)
plt.title('Average Human Capital')
plt.plot(histHid[200:])

plt.subplot(824)
plt.title('Unemployment Rate')
plt.plot(unemployed[200:])

plt.subplot(825)
plt.title('Fertility Rate')
plt.plot(fertilityRate[200:])

plt.subplot(826)
plt.title('Aggregate Demand')
plt.plot(histADemandOld[200:])

plt.subplot(827)
plt.title('Number of Firms')
plt.plot(nrFirms[200:])

plt.subplot(828)
plt.title('New Firms')
plt.plot(newFirms[200:])

plt.subplot(829)
plt.title('Bankrupt Firms')
plt.plot(bankruptFirms[200:])

plt.subplot(8,2,10)
plt.title('Log Average Worker Savings')
plt.plot(meanSavings[200:])

plt.subplot(8,2,11)
plt.title('Log Average Firm Savings')
plt.plot(meanFirmSavings[200:])

plt.subplot(8,2,12)
plt.title('Mean of Firm Debt')
plt.plot(meanFirmDebts[200:])

plt.subplot(8,2,13)
plt.title('Mean Wage')
plt.plot(meanWage[200:])

plt.subplot(8,2,14)
plt.title('Mean Worker Productivity')
plt.plot(meanWorkerProductivity[200:])

plt.subplot(8,2,15)
plt.title('Mean Firm Productivity')
plt.plot(meanFirmProductivity[200:])

plt.subplot(8,2,16)
plt.title('Mean Price')
plt.plot(meanPrice[200:])

plt.tight_layout()

plt.figure(figsize=(10,10))

plt.subplot(221)
plt.title('GDP Growth Rate')
plt.plot(GDPGrowth[200:])

plt.subplot(222)
plt.title('Price Growth Rate')
plt.ylim(-1, 2)
plt.plot(priceGrowth[200:])

plt.subplot(223)
plt.title('Wage Growth Rate')
plt.plot(wageGrowth[200:])

plt.subplot(224)
plt.title('GDP/AD')
plt.plot(GDPByAD[200:])

#plt.figure()
#plt.scatter(unemployed,priceGrowth)
#plt.xlabel('Unemployment rate')
#plt.ylabel('Price Growth')

plt.tight_layout()
plt.show()