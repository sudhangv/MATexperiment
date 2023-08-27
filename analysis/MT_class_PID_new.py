# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 2019

@author: Rstewart
"""
#%%
# This class' function is as a means of holding the data and settings
# tied to a single measurement run, as read by a .csv file containing
# data from the ADC and a .txt file containing all settings contained
# in MT_settings.py at the time of the run. 

# TODO: store starting and ending indices instead of phases instead of data itself?

import numpy as np
from numpy.testing._private.utils import assert_allclose
from scipy.optimize import curve_fit, minimize
from scipy.special import erf
import math
from scipy.integrate import quad
import scipy.special as sc
import os 

class MTdataHost:
	'''
	MTdataHost holds relevant data, parameters for a single decay rate measurement.

	Data is loaded using loadData or rawLoad, and relevant parameters (ex. recaptured fraction)
	may be set with function setAll and are attributes of the object.
	'''

	def __init__(self, sampleRate):
		'''
		Here we define variables that are defined for an instance of the class MTdataHost

		Variable: sampleRate - sample rate of the ADC (set in MT_decay_rate.py script)
		'''

		# sample rate of the ADC
		self.sampleRate = sampleRate

		# initialize arrays for raw time, voltage data
		self.voltage = np.array([], float)
		self.time = np.array([], float)
		self.extraTime = 0

		# time for various test sequences
		self.timeHold = 0
		self.timeBaseline = 0
		self.timeTest = 0
		self.timeLoad = 0
		self.timeF1 = 0

		# offset to data since data acquistion doesn't occur at the correct start time
		self.offset = 0

		# values of variables that will be used 
		self.baseVolt = 0
		self.BaseVoltErr = 0
		self.motSS = 0
		self.motSSErr = 0
		self.MTvolt = 0
		self.MTvoltErr = 0

		self.bkBool = False
		
		self.resultFile = 'resultDict.txt'
		

	def loadData(self, fileName, settingsName):
		'''
		loads data from a .csv file and setting file as written by MT_decay_rate.py

		Variables:
		fileName - location of .csv file containing raw time, voltage data

		settingsName - location of .txt file containing apparatus parameters set during measurement
		'''

		data = np.genfromtxt(fileName ,dtype=None, comments="#", delimiter=",", skip_header=1)
		self.settings = eval(open(settingsName, 'r').read())
		
		# set trap depth, holding time, as well as some other timing things
		self.timeHold = self.settings['wait_mtrap']
		self.timeBaseline = self.settings['wait_baseline']
		self.timeTest = self.settings['wait_image']
		self.timeLoad = self.settings['wait_Load']
		if self.settings['state'] == 1:
			self.timeF1 = self.settings['wait_hfine_pump_F1']
		else:
			self.timeF1 = self.settings['wait_hfine_pump_F2']
		self.timeCool = self.settings['wait_cool']
		self.extraTime = self.settings['extraTime']

		self.voltage = data[:,1]
		self.time = data[:,0] - min(data[:,0])

		# sets the offset to measurement sequence
		self.setOffset()

		# sets timing
		self.setTiming()
	
	def loadCATdata(self, fileName, settingsName):
		'''
		loads data from a .csv file and setting file as written by CATmeasurement.py

		Variables:
		fileName - location of .csv file containing raw time, voltage data

		settingsName - location of .txt file containing apparatus parameters set during measurement
		'''

		self.CATbkBool = False
  
		data = np.genfromtxt(fileName ,dtype=None, comments="#", delimiter=",", skip_header=1)
		self.settings = eval(open(settingsName, 'r').read())
		
		self.timeBaseline = self.settings['wait_baseline']
		self.timeTest = self.settings['wait_image']
		self.timeLoad = self.settings['wait_Load']
		self.timeDeload = self.settings['cat_deload_t']
		self.timeReload = self.settings['MOT_reload_t']

		self.voltage = data[:,1]
		self.time = data[:,0] - min(data[:,0])

		# sets the offset to measurement sequence
		self.setOffset()

		# sets timing
		self.setCATtiming()

	def backgroundData(self, fileName):
		data = np.genfromtxt(fileName ,dtype=None, comments="#", delimiter=",", skip_header=1)
		self.bkVoltage = data[:,1]
		self.bkTime = data[:,0] - min(data[:,0])
		self.bkBool = True

		## get background readings from beginning of data set
		indEnd = self.getTimeInd(self.offset) - 100

		self.bkbase = np.average(self.bkVoltage[:indEnd])
		self.bkFl = np.average(self.voltage[:indEnd])


		# measured ratio; V_fl = 0.328 * V_mon
		self.prop = 0.328
		
	def CATbackgroundData(self, fileName):
		data = np.genfromtxt(fileName ,dtype=None, comments="#", delimiter=",", skip_header=1)
		self.CATbkVoltage = data[:,1]
		self.CATbkTime = data[:,0] - min(data[:,0])
		self.CATbkBool = True    

	def rawLoad(self, data, settings):
		'''
		Loads data directly from a data array (thus may be used to analyse data directly in MT_decay_rate.py script)

		Variables:
		data - array containing time, voltage data; first column are times, second are voltages

		settings - dictionary containing apparatus parameters
		'''

		self.settings = settings
		
		# set trap depth, holding time, as well as some other timing things
		self.timeHold = self.settings['wait_mtrap']
		self.timeBaseline = self.settings['wait_baseline']
		self.timeTest = self.settings['wait_image']
		self.timeLoad = self.settings['wait_Load']
		if self.settings['state'] == 1:
			self.timeF1 = self.settings['wait_hfine_pump_F1']
		else:
			self.timeF1 = self.settings['wait_hfine_pump_F2']
		self.timeCool = self.settings['wait_cool']
		self.extraTime = self.settings['extraTime']

		self.voltage = data[:,1]
		self.time = data[:,0] - min(data[:,0])

		# sets the offset to measurement sequence
		self.setOffset()

		# sets timing
		self.setTiming()
		
	def rawBackgroundData(self, data):
		self.bkVoltage = data[:,1]
		self.bkTime = data[:,0] - min(data[:,0])
		self.bkBool = True

		## get background readings from beginning of data set
		indEnd = self.getTimeInd(self.offset) - 100

		self.bkbase = np.average(self.bkVoltage[:indEnd])
		self.bkFl = np.average(self.voltage[:indEnd])


		# measured ratio; V_fl = 0.328 * V_mon
		self.prop = 0.328


	def setOffset(self):
		'''
		Sets the offset to all times in the data, based upon where voltage jumps occur
		and the time the background values are recorded at
		'''

		# search within the first 0.5 seconds
		endInd = self.getTimeInd(15)

		testInd = self.getTimeInd(0.005)

		# find value of voltage at beginning of data
		compar = np.average(self.voltage[:testInd])
		self.noLightBackground = compar
		#print(f"\n \n Compar = {compar} \n \n") 
		# find where voltage is greater than any typical background value
		if compar < 0:
			offsetPts = np.where((self.voltage[testInd:endInd] - compar) > abs(2*compar))
		else:
			offsetPts = np.where(self.voltage[testInd:endInd] > 2*compar)

		# try to set offset
		try:
			self.offset = self.time[offsetPts[0][0]]
		# raise error if not found
		except IndexError:
			raise UserWarning('Offset not found.')
		

	def getAverage(self, data):
		values = np.average(data)
		error = np.std(data)/np.sqrt(len(data))
		return values, error
	
	def getTimeInd(self, t):
		'''
		Converts passed time t to array index based upon data collection rate of the ADC
		'''

		return int(np.floor(t*self.sampleRate))

	def setTiming(self):
		'''
		Sets timing of various stages, defined by the time of each stage in the loaded
		settings file
		'''

		self.tLoad = self.offset + self.timeBaseline
		if self.settings['state'] == 2:
			self.tMT = self.timeCool + self.tLoad + self.timeLoad + self.timeF1 + self.timeHold +0.2
			if self.settings['clearbool']:
				self.tMT += self.extraTime
		else:
			self.tMT = self.timeCool + self.tLoad + self.timeLoad + self.timeF1 + self.timeHold
			
		if self.settings['wallbool']:
			self.tMT += 0
		self.tBaseline = self.tMT + self.timeTest + self.timeTest
		
	def setCATtiming(self):
		self.tCATbackground = self.offset 
		self.tLoad = self.offset + self.timeBaseline + self.timeBaseline + + self.timeBaseline 
		
		self.tDeload = self.tLoad + self.timeLoad 
		self.tReload = self.tDeload + self.timeDeload
			
		#else:
		#    self.tMT = self.timeCool + self.tLoad + self.timeLoad + self.timeF1 + self.timeHold
			
		self.tBaseline = self.tReload + self.timeReload + self.timeTest
		
	def setBaseline(self, timeBuffer):
		'''
		Sets the baseline voltage value from the scattered light due to both
		pump and repump
		'''

		tStart = self.tBaseline + timeBuffer*10 # 10
		
		tEnd = tStart + self.timeBaseline - timeBuffer*10
		

		startInd, endInd = self.getTimeInd(tStart), self.getTimeInd(tEnd)
		
						
		for i in range(startInd, endInd):
			if (self.voltage[i] > 6*0.01):
				tStart = self.time[i-1]
				break
		startInd = self.getTimeInd(tStart + 0.1)

		avValue, avErr = self.getAverage(self.voltage[startInd:endInd])
		self.base1, self.base2 = startInd, endInd

		self.baseTime, self.baseVoltage = self.time[startInd:endInd], [avValue for item in self.voltage[startInd:endInd]]

		self.baseVolt = avValue
		self.BaseVoltErr = avErr

		# define a typical standard deviation for the voltage uncertainty
		tEnd2 = self.getTimeInd(tStart + 0.005) 
		# take std over 5 ms
		self.std = 2*np.std(self.voltage[startInd:endInd])
		
		if self.bkBool:
			avMon, avMonerr = self.getAverage(self.bkVoltage[self.base1:self.base2])
			self.prop = (self.baseVolt - self.bkFl) / (avMon - self.bkbase)


	def setCATbaseline(self, timeBuffer):
		'''
		Sets the voltage value from the scattered light due to the catalysis beam
		'''

		tStart = self.tCATbackground + timeBuffer*10 # 10
		
		tEnd = tStart + self.timeBaseline - timeBuffer*10
		
		startInd, endInd = self.getTimeInd(tStart), self.getTimeInd(tEnd)
		
						
		for i in range(startInd, endInd):
			if (self.voltage[i] > 2.5*0.01):   # TODO: 
				tStart = self.time[i-1]
				break
		startInd = self.getTimeInd(tStart + 0.1)

		avValue, avErr = self.getAverage(self.voltage[startInd:endInd])
		self.CATbackground1, self.CATbackground2 = startInd, endInd

		self.CATbackgroundTime, self.CATbackgroundVoltage = self.time[startInd:endInd], [avValue for item in self.voltage[startInd:endInd]]

		self.CATbackgroundVolt = avValue - self.noLightBackground
		self.CATbackgroundVoltErr = avErr
		
		if self.CATbkBool:
			testInd = self.getTimeInd(0.005)
			compar = np.average(self.voltage[:testInd])
			
			self.CATnoLightBackground = compar        
			self.CATbkAvg = np.mean(self.CATbkVoltage[startInd:endInd]) - self.CATnoLightBackground
			
			self.CATprop = self.CATbackgroundVolt/ self.CATbkAvg

	def fitLinear(self, time, voltage):

		def linear(x, m, b):
			return m*x + b

		time = time - time[0]
		popt, pcov = curve_fit(linear,
								time, 
								voltage, 
								sigma = [self.std]*len(voltage), 
								absolute_sigma=True, 
								maxfev=10**5)
		fitUnc = np.sqrt(np.diag(pcov))

		return popt, fitUnc, linear(time, *popt)
	
	def fitExp(self, time, voltage):
		
		# def exp(t, V_0, gamma):
		#     return V_0 + (self.motA - V_0)*(1 - np.exp(-gamma*t))
		def exp(t, V_0, A, gamma):
			return V_0 + A*(1 - np.exp(-gamma*t))
		
		time = time - time[0]
		popt, pcov = curve_fit(exp,
								time, 
								voltage, 
								sigma = [self.std]*len(voltage), 
								absolute_sigma=True, 
								maxfev=10**5)
		fitUnc = np.sqrt(np.diag(pcov))

		return popt, fitUnc, exp(time, *popt)
		

	def setLoading(self, timeBuffer):
		'''
		Sets the MOT loading voltage curve, as well as the steady state MOT value
		'''

		# sets the start and end time based upon the timing of the previous sequences
		tStart = self.tLoad + 5*timeBuffer

		tEnd1 = tStart + self.timeLoad - 10*timeBuffer
		tEnd2 = tStart + self.timeLoad + 10*timeBuffer

		end1, end2 = self.getTimeInd(tEnd1), self.getTimeInd(tEnd2)

		# for i in range(end1, end2):
		#     if self.voltage[i] < self.baseVolt:
		#         endInd = i-1
		#         break
		
		startInd = self.getTimeInd(tStart)
		
		# TODO: correct how the ending indice is found
		endInd = end1 + np.argmax(abs(self.voltage[end1:end2]))-1
		
		self.loadingVoltage = self.voltage[startInd:endInd]
		self.loadingTime = self.time[startInd:endInd]

		# fit loading rate to exponential model
		def exp(t, V_0, A, gamma):
			return V_0 + A*(1 - np.exp(-gamma*t))

		# def constVexp(t, V_0, betaN, gamma, R):
		#     gammaEff = gamma + 2*betaN
		#     xi = betaN/(gamma+ betaN)
		#     return V_0 + (R/(gamma+betaN))*(1 - np.exp(-gammaEff*t))/(1 + xi*np.exp(-gammaEff*t))
		if self.bkBool:
			popt,pcov = curve_fit(exp,
									self.loadingTime - self.loadingTime[0], 
									(self.loadingVoltage - self.bkFl) - (self.bkVoltage[startInd:endInd] - self.bkbase)*self.prop, 
									sigma = [self.std]*len(self.loadingVoltage), 
									absolute_sigma=True, 
									maxfev = 10**6,
									p0 = [0, 0.5, 0.2])
		else:
			popt,pcov = curve_fit(exp,
						self.loadingTime - self.loadingTime[0], 
						self.loadingVoltage, 
						sigma = [self.std]*len(self.loadingVoltage), 
						absolute_sigma=True, 
						maxfev = 10**5,
						p0 = [0.05, 0.3, 0.5])
		fitUnc = np.sqrt(np.diag(pcov))

		# set MOT loss rate constant
		self.motA = popt[1]
		self.motR, self.motRErr = popt[2], fitUnc[2]
		self.motFitR = popt[1]*popt[2]
		self.motFitRErr = self.motFitR*np.sqrt((fitUnc[2]/popt[2])**2 + (fitUnc[1]/popt[1])**2)
		self.motFit = exp(self.loadingTime - self.loadingTime[0], *popt)
		
	

		# taking last point of fitting exponential to be the MOT SS value
		if self.bkBool:
			self.motSS = exp(self.loadingTime - self.loadingTime[0], *popt)[-1]
			#self.motSS = np.average(((self.loadingVoltage - self.bkFl) - (self.bkVoltage[startInd:endInd] - self.bkbase)*self.prop)[-100:])
			self.motSSErr = (fitUnc[0]**2 + fitUnc[1]**2)**(0.5)
			if self.motSSErr / self.motSS > 0.01:
				self.motSSErr = self.motSS*0.01
		else:
			self.motSS = exp(self.loadingTime - self.loadingTime[0], *popt)[-1] - self.baseVolt
			#self.motSS = np.average(self.loadingVoltage[-100:])- self.baseVolt
			self.motSSErr = (fitUnc[0]**2 + fitUnc[1]**2 + self.BaseVoltErr**2)**(0.5)

	def setInitialLoad(self, timeBuffer):
		'''
		Sets the initial loading rate R (V/s) based upon fitting start of loading curve
		to a linear model
		'''

		# fit over 1/10 of the associated half-life of the MOT loading rate curve 
		tStart = self.tLoad +4*timeBuffer

		# if 1/10 of the associated half-life is too large, then fit over a range 0.5 s
		# this only seems to have an influence on very small MOTs
		if (1/self.motR)*0.1 < 1:
			tEnd = tStart + 1/self.motR*0.1
		else:
			tEnd = tStart + 2 # use 0.1 for high, 0.5 for low
	
		startInd, endInd = self.getTimeInd(tStart), self.getTimeInd(tEnd)
		self.initStartInd, self.initEndInd = startInd, endInd
		
		if self.bkBool:
			vals = self.fitLinear(self.time[startInd:endInd], (self.voltage[startInd:endInd] - self.bkFl)- (self.bkVoltage[startInd:endInd] - self.bkbase) * self.prop)
		else:
			vals = self.fitLinear(self.time[startInd:endInd], self.voltage[startInd:endInd])

		self.initMOTR, self.initMOTRErr = vals[0][0], vals[1][0]
		self.initTime = self.time[startInd:endInd]
		self.initVolt = vals[2]

		return vals, self.time[startInd:endInd]
		
	def setMToffset(self, tStart, tEnd):
		'''
		Sets the time at which the MT ends by finding where voltage increases to a value
		well above background
		'''

		indStart, indEnd = self.getTimeInd(tStart), self.getTimeInd(tEnd)
		endTime = tStart
		
		for i in range(indStart, indEnd):
			if (np.average(self.voltage[i:i+1]) > self.baseVolt-self.std):
				endTime = self.time[i]
				break
		return endTime

	def setMTExEnd(self, tStart, tEnd):
		'''
		Sets end of MOT loading curve by finding where voltage decreases to a value well
		below the scattered light
		'''

		indStart, indEnd = self.getTimeInd(tStart), self.getTimeInd(tEnd)
		endTime = tStart

		for i in range(indStart, indEnd):
			if (self.voltage[i] < (self.baseVolt/2)): #
				endTime = self.time[i-1]
				break
		return endTime



	def setMTVolt(self, timeBuffer):
		'''
		Sets the extrapolated MT voltage by fitting to a linear function
		'''

		# find end point of MT stage
		tMTEnd = self.setMToffset(self.tMT - 50*timeBuffer, self.tMT + 4000*timeBuffer)

		tStart = tMTEnd + 0*timeBuffer
		tEnd = tStart + self.timeTest - 2*timeBuffer
		tEnd = self.setMTExEnd(tStart + 10*timeBuffer, tStart + self.timeTest + 10*timeBuffer)-10*timeBuffer-0.85 # added this to reduce range of fit, 90 for fast, 10 for slow, 125 for fastest
		indStart, indEnd = self.getTimeInd(tStart+10*timeBuffer), self.getTimeInd(tEnd) # 20 for fast, 10 for fastest

		# fit to linear function
		if self.bkBool:
			fitVals, fitUnc, voltVals = self.fitLinear(self.time[indStart:indEnd], (self.voltage[indStart:indEnd] - self.bkFl) - (self.bkVoltage[indStart:indEnd] - self.bkbase) * self.prop)
		else:
			fitVals, fitUnc, voltVals = self.fitLinear(self.time[indStart:indEnd], self.voltage[indStart:indEnd])

		self.linVoltage, self.linTime = voltVals, self.time[indStart:indEnd]
		timeDiff = (tStart+10*timeBuffer - tMTEnd)

		# extrapolate back to when MT stage ends
		if self.bkBool:
			self.MTvolt = fitVals[0]*(-timeDiff) + fitVals[1]

			self.MTvoltpt, self.MTvoltT = fitVals[0]*(-timeDiff) + fitVals[1], tMTEnd
			self.MTvoltErr = ((fitUnc[0]*timeDiff)**2 + fitUnc[1]**2)**(0.5)
			self.MTvoltSlope = fitVals[0]
		else:  
			self.MTvolt = fitVals[0]*(-timeDiff) + fitVals[1] - self.baseVolt

			self.MTvoltpt, self.MTvoltT = fitVals[0]*(-timeDiff) + fitVals[1], tMTEnd
			self.MTvoltErr = ((fitUnc[0]*timeDiff)**2 + fitUnc[1]**2 + self.BaseVoltErr**2)**(0.5)
			self.MTvoltSlope = fitVals[0]
			
	def setMTVolt_exp(self, timeBuffer):
		tMTEnd = self.setMToffset(self.tMT - 50*timeBuffer, self.tMT + 4000*timeBuffer)

		tStart = tMTEnd + 0*timeBuffer
		tEnd = tStart + self.timeTest - 2*timeBuffer
		tEnd = self.setMTExEnd(tStart + 10*timeBuffer, tStart + self.timeTest + 10*timeBuffer)- 10*timeBuffer # added this to reduce range of fit, 80 for fast, 10 for slow
		indStart, indEnd = self.getTimeInd(tStart+20*timeBuffer), self.getTimeInd(tEnd) # 10 for fast

		# fit to linear function
		if self.bkBool:
			fitVals, fitUnc, voltVals = self.fitExp(self.time[indStart:indEnd], (self.voltage[indStart:indEnd] - self.bkFl) - (self.bkVoltage[indStart:indEnd] - self.bkbase) * self.prop)
		else:
			fitVals, fitUnc, voltVals = self.fitExp(self.time[indStart:indEnd], self.voltage[indStart:indEnd])

		self.linVoltage, self.linTime = voltVals, self.time[indStart:indEnd]
		tMTEnd+=0*timeBuffer
		timeDiff = (tStart+20*timeBuffer - tMTEnd)
		
		# def exp(t, V_0, gamma):
		#     return V_0 + (self.motA - V_0)*(1 - np.exp(-gamma*t))
		
		def exp(t, V_0, A, gamma):
			return V_0 + A*(1 - np.exp(-gamma*t))
		
		
		#print(fitVals)
		# extrapolate back to when MT stage ends
		if self.bkBool:
			#self.MTvolt = fitVals[0]*(-timeDiff) + fitVals[1]
			self.MTvolt = exp(-timeDiff, *fitVals)

			self.MTvoltpt, self.MTvoltT = exp(-timeDiff, *fitVals), tMTEnd
			self.MTvoltErr = ((fitUnc[0])**2 + fitUnc[1]**2)**(0.5)/20
			self.MTvoltSlope = fitVals[0]
		else:  
			self.MTvolt = exp(-timeDiff, *fitVals) - self.baseVolt

			self.MTvoltpt, self.MTvoltT = self.MTvolt + self.baseVolt, tMTEnd
			self.MTvoltErr = ((fitUnc[0]*timeDiff)**2 + fitUnc[1]**2 + self.BaseVoltErr**2)**(0.5)
	
	def setDeloadEnd(self, timeBuffer):
		tStart = self.tReload - 100*timeBuffer
		tEnd = self.tReload + 100*timeBuffer
		
		indStart = self.getTimeInd(tStart)
		indEnd = self.getTimeInd(tEnd)
		
		tDeloadEnd = self.tReload   # if don't find offset, use settings for timing
		# TODO: there definitely are better ways to do this
		for i in range(indStart, indEnd):
			if abs(self.voltage[i+1]-self.voltage[i]) > 5*self.std:
				tDeloadEnd = self.time[i+1]
		
		
		self.tDeloadEnd = tDeloadEnd
		
		return tDeloadEnd
		
		
	def setReloadVolt(self, timeBuffer):
		
		linear_fit = False
		tStart = self.setDeloadEnd(timeBuffer)
		timeDiff = 20*timeBuffer
		
		tEnd = tStart + self.timeReload - 20*timeBuffer
		#tEnd = self.setMTExEnd(tStart + 10*timeBuffer, tStart + self.timeTest + 10*timeBuffer)- 10*timeBuffer # added this to reduce range of fit, 80 for fast, 10 for slow
		indStart, indEnd = self.getTimeInd(tStart+timeDiff), self.getTimeInd(tEnd) # 10 for fast
		
		self.reloadStartInd = indStart
		self.reloadEndInd = indEnd

		reloadVoltage =  self.voltage[indStart:indEnd]
		reloadTime =  self.time[indStart:indEnd]
		

		self.reloadVoltage = reloadVoltage        
		self.reloadTime = reloadTime
		
		fitVals, fitUnc, voltVals = self.fitExp(reloadTime, reloadVoltage)

		# TODO: compare uncertainties between the two kind of fits before chosing one?
		if max(fitUnc/fitVals) > 0.5:
			fitVals, fitUnc, voltVals = self.fitLinear(reloadTime, reloadVoltage)
			linear_fit = True
		self.reloadFitVoltage = voltVals
		


		def exp(t, V_0, A, gamma):
			return V_0 + A*(1 - np.exp(-gamma*t))
		
		def linear(x, m, b):
			return m*x + b
		# extrapolate back to when deload stage ends
		
		if linear_fit:
			self.reloadVolt = linear(-timeDiff, *fitVals) - self.baseVolt
			self.reloadVoltErr = ((fitUnc[0]*timeDiff)**2 + fitUnc[1]**2 + self.BaseVoltErr**2)**(0.5)
		else:    
			self.reloadVolt = exp(-timeDiff, *fitVals) - self.baseVolt
			self.reloadVoltErr = (fitUnc[0]**2 + ((1 - np.exp(-fitVals[2]*timeDiff))*fitUnc[1])**2 + ((fitVals[1]*timeDiff*np.exp(-fitVals[2]*timeDiff)) * fitUnc[2])**2)**0.5
			
		self.reloadVoltpt, self.reloadVoltT = self.reloadVolt + self.baseVolt, tStart
		self.linear_reload_fit = linear_fit

	def setDeloading(self, timeBuffer):
		
		# sets the start and end time based upon the timing of the previous sequences
		tStart = self.tDeload + 15*timeBuffer

		tEnd1 = tStart + self.timeDeload - 10*timeBuffer
		tEnd2 = tStart + self.timeDeload + 10*timeBuffer

		end1, end2 = self.getTimeInd(tEnd1), self.getTimeInd(tEnd2)

		startInd = self.getTimeInd(tStart)
		endInd = end1 
		
		self.deloadVoltage = self.voltage[startInd:endInd]
		self.deloadTime = self.time[startInd:endInd]

		if self.CATbkBool:
			self.deloadVoltage = self.voltage[startInd:endInd] - self.CATprop*self.CATbkVoltage[startInd:endInd] - self.baseVolt
		# fit loading rate to exponential model
		# def deloadModel(t, gamma, betaf, R, N0):
		#     t1 = np.sqrt(4*R*betaf + gamma**2)
		#     t2 = gamma + 2*betaf*N0
		#     result = -gamma/(2*betaf) + t1*np.tanh(0.5*t*t1+np.atanh(t2/t1))/(2*betaf)
		#     return result

		# # TODO from here onwards
		# popt,pcov = curve_fit(deloadModel,
		#             self.deloadingTime - self.deloadingTime[0], 
		#             self.deloadingVoltage, 
		#             sigma = [self.std]*len(self.deloadingVoltage), 
		#             absolute_sigma=True, 
		#             maxfev = 10**5,
		#             p0 = [0.05, 0.3, 0.5])
		# fitUnc = np.sqrt(np.diag(pcov))

		# # set deload rate constants
		# self.motA = popt[1]
		# self.motR, self.motRErr = popt[2], fitUnc[2]
		# self.motFitR = popt[1]*popt[2]
		# self.motFitRErr = self.motFitR*np.sqrt((fitUnc[2]/popt[2])**2 + (fitUnc[1]/popt[1])**2)
		# self.motFit = exp(self.loadingTime - self.loadingTime[0], *popt)
		
	

		# # taking last point of fitting exponential to be the MOT SS value
		# if self.bkBool:
		#     self.motSS = exp(self.loadingTime - self.loadingTime[0], *popt)[-1]
		#     #self.motSS = np.average(((self.loadingVoltage - self.bkFl) - (self.bkVoltage[startInd:endInd] - self.bkbase)*self.prop)[-100:])
		#     self.motSSErr = (fitUnc[0]**2 + fitUnc[1]**2)**(0.5)
		#     if self.motSSErr / self.motSS > 0.01:
		#         self.motSSErr = self.motSS*0.01
		# else:
		#     self.motSS = exp(self.loadingTime - self.loadingTime[0], *popt)[-1] - self.baseVolt
		#     #self.motSS = np.average(self.loadingVoltage[-100:])- self.baseVolt
		#     self.motSSErr = (fitUnc[0]**2 + fitUnc[1]**2 + self.BaseVoltErr**2)**(0.5)
	def setAll(self, timeBuffer):
		'''
		Sets all of the relevant parameters, as well as the 'ratio' variable, which is the recaptured fraction
		'''
		
		self.setBaseline(timeBuffer)

		self.setLoading(timeBuffer)
		self.setMTVolt_exp(timeBuffer)

		self.initFit, self.initX = self.setInitialLoad(timeBuffer)

		# recaptured fraction
		self.ratio = self.MTvolt / self.motSS
		# if self.ratio < 0:
		#     self.ratio = 0.0000001
		self.ratioErr = self.ratio * ((self.MTvoltErr/self.MTvolt)**2 + (self.motSSErr/self.motSS)**2)**(0.5)
		if abs(self.ratioErr / self.ratio) > 0.015:
			self.ratioErr = abs(0.015*self.ratio)
		if self.ratioErr < 0.001:
			self.ratioErr = 0.001
		# elif abs(self.ratioErr) > 0.01:
		#     self.ratioErr = abs(0.01)
		if self.bkBool:
			self.voltage = self.voltage - (self.bkFl + (self.bkVoltage - self.bkbase) * self.prop)
			self.baseVoltage = np.array([0 for item in self.baseVoltage])
		if self.ratio < 0:
			self.ratio = 0
		print('File loaded: RFmin = {} MHz, t_mt = {:.3f} s.'.format(self.settings['fmin'], self.settings['wait_mtrap']))
	
	def setAllCAT(self, timeBuffer):
		
		self.setCATbaseline(timeBuffer)
		self.setBaseline(timeBuffer)

		self.setLoading(timeBuffer)
		self.setDeloading(timeBuffer)   # TODO: currently just stores the deloading times and voltages
		self.setReloadVolt(timeBuffer)

		self.initFit, self.initX = self.setInitialLoad(timeBuffer)

		# steady state ratio fraction
		self.ratio = self.reloadVolt / self.motSS

		self.ratioErr = self.ratio * ((self.reloadVoltErr/self.reloadVolt)**2 + (self.motSSErr/self.motSS)**2)**(0.5)
		
		# if abs(self.ratioErr / self.ratio) > 0.1:
		#     self.ratioErr = abs(0.015*self.ratio)
		if self.ratioErr < 0.001:
			self.ratioErr = 0.001
		if self.ratio < 0:
			self.ratio = 0
			
			# TODO: this information is useless
		print('File loaded: RFmin = {} MHz, t_mt = {:.3f} s.'.format(self.settings['fmin'], self.settings['wait_mtrap']))
		
	def getResults(self, dirName, store=True):
		"""_summary_

		Args:
			dirName (_str_): directory in which to store the fitted results
		"""
		dataDict = {}
		for key,value in self.__dict__.items():
			if not hasattr(value, '__iter__') and not callable(value):  # don't store any iterables and functions
				dataDict[key] = value
	
		if store:
			with open(os.path.join(dirName, self.resultFile), 'w') as f:
				f.write(str(dataDict))

		return dataDict
	
	def storeFits(self, dirName, combined=True, separate=False):
		import matplotlib.pyplot as plt
		
		if combined:
			plt.scatter(self.time, self.voltage, s=0.1)
			plt.plot(self.reloadTime,self.reloadFitVoltage, c='orange')
			plt.plot(self.loadingTime,self.motFit, c='red')
			plt.plot(self.baseTime,self.baseVoltage, c='yellow')
			plt.plot(self.CATbackgroundTime,self.CATbackgroundVoltage, c='pink')
			
			plt.savefig(os.path.join(dirName,'fits.png'), dpi=400)
			plt.close()
			
		if separate:
			plt.plot(self.loadingTime,self.loadingVoltage)
			plt.plot(self.loadingTime,self.motFit)
			plt.scatter(self.loadingTime[-1],(self.motSS+self.baseVolt), c='red')            
			plt.savefig(os.path.join(dirName,'loadFit.png'), dpi=200)
			plt.close()
			
			plt.plot(self.reloadTime,self.reloadVoltage)
			plt.plot(self.reloadTime,self.reloadFitVoltage)
			plt.scatter(self.reloadVoltT,self.reloadVoltpt, c='red')            
			plt.savefig(os.path.join(dirName,'reloadFit.png'), dpi=200)
			plt.close()
			
			plt.plot(self.CATbackgroundTime,self.voltage[self.CATbackground1:self.CATbackground2])
			plt.plot(self.CATbackgroundTime,self.CATbackgroundVoltage)
			plt.savefig(os.path.join(dirName,'CATbackgroundFit.png'), dpi=200)
			plt.close()
			
			plt.plot(self.baseTime,self.voltage[self.base1:self.base2])
			plt.plot(self.baseTime,self.baseVoltage)
			plt.savefig(os.path.join(dirName,'baselineFit.png'), dpi=200)
			plt.close()
			
			plt.plot(self.deloadTime,self.deloadVoltage)
			plt.savefig(os.path.join(dirName,'deloadPhase.png'), dpi=200)
			plt.close()

			ind1 = int(self.initTime[0]*2000)
			ind2 = ind1 + len(self.initTime)
			plt.plot(self.initTime, self.voltage[ind1:ind2+1][:len(self.initTime)])
			plt.plot(self.initTime, self.initFit[2])
			plt.savefig(os.path.join(dirName,'initFit.png'), dpi=200)
			plt.close()
		
		
class fitter:
	'''
	Fitter holds methods and functions relevant to fitting data after the recaptured fraction has been
	determined for each trap depth and MT holding time.
	'''

	def __init__(self, state, iso=85):
		self.h = 4.136*10**(-15) # eV seconds
		self.kB = 8.617*10**(-5) # eV/K
		self.state = state
		self.iso = iso

	def findNearestInd(self, value, array):
		'''
		Returns indice of element in array closest to value
		'''

		return np.argmin(abs(array - value))
	

	def maxTrapDepth(self, current):
		B = 55/2 # G/cmA
		mu_B = 9.274*10**(-28) # J/G
		kB = 1.3807*10**(-23) # J/K
		g_F = 1/2 # G_F = 1/2, 87, G_F = 1/3 85
		if self.iso == 85:
			g_F = 1/3
		m_F = self.state
		dist = 0.5
		
		Uradial = (mu_B * g_F * m_F * B * current * dist)/kB * 1000  # in mK
		sag = 0.5128
		if self.iso == 85:
			sag = 0.501
		Uaxial = 2 * Uradial - sag
	
		return min(Uaxial, Uradial)

	def gravCorr(self, I, I_o, frequency):
		'''
		Calculates RF minimum frequency corrected for grav. sag using the minimum current
		'''
		value = self.state*self.h*frequency*(1 - I_o/I)
		maxPossible = self.maxTrapDepth(I)
		try:
			len(frequency)
			t = []
			for item in frequency:
				t.append(min(maxPossible, self.state*self.h*item*(1 - I_o/I)))
			return np.array(t)

		except:
			return min(maxPossible, value)

	def Expfit(self, xdata, ydata, ydataErr):

		def exp(t, A, gamma):
			return A*np.exp(-gamma*t)

		popt, pcov = curve_fit(exp, xdata, ydata, sigma=ydataErr, p0 = [ydata[0], 0.2], absolute_sigma=True, maxfev = 10**6)

		xplot = np.linspace(0, max(xdata), 20)

		return popt, np.sqrt(np.diag(pcov)), xplot, exp(xplot, *popt)

	def Linfit(self, xdata, ydata, yerr):
	
		def linear(x, m, b):
			return m*x + b

		popt, pcov = curve_fit(linear, xdata, ydata, sigma = yerr, absolute_sigma = True, maxfev=10**5)
	
		return popt, np.sqrt(np.diag(pcov)), xdata, linear(xdata, *popt)


	def tempFit(self, RFmins, frac, fracErr, minCurrent, Current):
		'''
		Fits data of recaptured fraction as a function of RF knife minimum frequency,
		correcting for gravitation sag. 
		
		Variables:
		RFmins - array containing minimum RF knife frequencies used (in MHz)

		frac - array containing corresponding recaptured fraction values

		fracErr - array containing recaptured fraction uncertainties

		minCurrent - minimum current to sustain trapped state against gravity (A)

		Current - current used to trap atoms during MT stage (A)
		'''

		def MBEDist(E, T, B, C):
			# Energy is always positive (this works better than just setting bounds for curve_fit)
			E = E - C
			ind_neg = np.where(E < 0)[0]
			E[ind_neg] = 0
			
			# max trap depth - CDF levels off there
			Emax = self.maxTrapDepth(Current)/1000 * self.kB

			# Temperature is always positive (same reason as above)
			T = abs(T)
			
			t1 = erf(np.sqrt(E / (self.kB*T)))
			t2 = 2/np.sqrt(math.pi)*np.exp(-E/(self.kB*T)) * np.sqrt(E / (self.kB*T))
			
			t1max = erf(np.sqrt(Emax / (self.kB*T)))
			t2max = 2/np.sqrt(math.pi)*np.exp(-Emax/(self.kB*T)) * np.sqrt(Emax / (self.kB*T))
			
			ind_above = np.where(E > Emax)[0]
			ValMax = B*(t1max - t2max)
	
			Values = B*(t1 - t2)
			#Values[ind_above] = ValMax
			return Values

		# formulate guess on parameters
		B_guess = max(frac)
		# here, 0.609 is the value that the unscaled MBEdist is at E = T
		T_guess = RFmins[self.findNearestInd(B_guess*0.609, frac)] * self.h * 10**6 / self.kB  # in microKelvin
		C_guess = RFmins[np.where(frac > 0.005)[0][0]] * self.h * 10**6  #in J
		C_guess = 0
		#print([T_guess, B_guess, C_guess])
		# correct for gravitational sag
		RFmins_corr = self.gravCorr(Current, minCurrent, RFmins*10**6)

		popt, pcov = curve_fit(MBEDist, RFmins_corr, frac, sigma = fracErr, p0 = [T_guess, B_guess, C_guess], absolute_sigma=True, maxfev = 10**6)#, bounds = ([0, 0, 0], [400e-6, 0.6, np.inf]))
		fitUnc = np.sqrt(np.diag(pcov))

		popt[0] = abs(popt[0])

		RFminsplot = np.linspace(0, max(RFmins_corr), max(100, len(RFmins_corr)))

		return popt, fitUnc, MBEDist(RFminsplot, *popt), RFmins_corr/self.kB*1000, RFminsplot/self.kB * 1000 # in MHz # in MHz
	
	def tempFit_JB(self, RFmins, frac, fracErr, minCurrent, Current):
		
		def cdf_JB(fmin, T, C):
			
			return 0
		
		
		return 0

	def calculateU(self, RFmins, temperature, RF_offset, current, minCurrent):
		'''
		Calulates the magnetic trap depth using knowledge of the temperature, RF offset,
		as well as the current used to trap. 
		
		Variables:
		RFmins - array containing RF knife minimum frequencies (MHz)

		temperature - temperature of the ensemble determined by tempFit

		RF_offset - RF offset determined by tempFit

		minCurrent - minimum current to sustain trapped state against gravity (A)

		Current - current used to trap atoms during MT stage (A)
		'''

		RFmins_corr = self.gravCorr(current, minCurrent, RFmins)

		E_RF_no_offset = RFmins_corr - RF_offset

		# calculates correction to trap depth due to finite mean energy of trapped atoms
		def Ucorrection(Ec, T):
			a = Ec / (self.kB*T)
			
			# analytical expressions for correction terms
			N1 = 3/2*erf(np.sqrt(a))
			N2 = 1/np.sqrt(math.pi)*np.exp(-a)*np.sqrt(a)*(2*a+3)

			D1 = erf(np.sqrt(a))
			D2 = 2/np.sqrt(math.pi)*np.exp(-a) * np.sqrt(a)


			return self.kB*T*(N1-N2)/(D1-D2)

		return RFmins_corr - (RF_offset + Ucorrection(E_RF_no_offset, temperature)) # in eV

	def pQDU(self, U, U_d, coefficients):
		'''
		Universal function describing dependence of trap loss rate on trap depth U

		Variables:
		U - trap depth

		U_d - trap depth corresponding to quantum diffractive energy

		coefficients - coefficients in power expansion in powers of U/U_d
		'''

		UUd = U/U_d
		UUds = np.array([UUd**i for i in range(0,len(coefficients))])
		pqdu6 = np.dot(coefficients, UUds)

		return pqdu6

	def lossSec(self, U, sigmaV, nRb): # U must be in eV
		'''
		Loss rate as a function of trap depth U.

		Variables:
		U - trap depths

		sigmaV - fitting parameter; collisional cross-section. Used to determine U_d

		nRb - fitting parameter; background rubidium density
		'''

		U_d = self.Ud(sigmaV)[0]

		#betaJ = [1, -0.6754, 0.4992, -0.228, 0.1165,  -0.0321, 0.00413] # in order of degree 1 to degree 6
		betaJ = [1, -0.6730, 0.477, -0.228, 0.0703, -0.0123, 0.0009]
		#betaJ = [1, -0.6730, 0.477, -0.228]

		return 10**(-3)*nRb*sigmaV*(self.pQDU(U, U_d, betaJ))      


	def Ud(self, sigmaV):
		'''
		Calculate Ud based on value of collisional cross-section sigmaV.
		'''

		T = 294
		m_t = 9.00750348*10**(-7) # mass of Rb87 (in eV/(m/s)^2)
		m_bg = 8.80048847*10**(-7) # mass of Rb85 (in eV/(m/s)^2)
		#m_bg = 2.09*10**(-8) # mass of H2

		# most probable speed of impinging particles
		v_p = np.sqrt((2 * self.kB * T)/(m_bg))

		U_d = (4 * math.pi * (self.h/(2*math.pi))**2 * v_p)/(m_t * sigmaV * 10**(-15))

		return U_d, v_p # in eV, m/s


	def fitLossSec(self, U, G, GErr):
		'''
		Fit data of decay rate and corresponding trap depths to extract collisional cross section
		and background Rb density.

		Variables:
		U - trap depths (eV)

		G - decay rates (Hz)
		GErr - associated decay rate uncertainties (Hz)
		'''
		#p0 = [2*10**(-15), 9*10**(12)]
		popt, pcov = curve_fit(self.lossSec, U, G, sigma = GErr, p0 = [6.45, 150], absolute_sigma=True, maxfev=10**7)

		Uplot = np.linspace(min(U), max(U), 100)

		return popt, np.sqrt(np.diag(pcov)), Uplot/self.kB*1000, self.lossSec(Uplot, *popt)

	def fitLossSecnRb(self, U, G, GErr, sigmaV): # if sigma is known by some other method, can determine nRb
		
		U_d = self.Ud(sigmaV)[0]

		betaJ = [1, -0.6730, 0.477, -0.228, 0.0703, -0.0123, 0.0009]
		#betaJ = [1, -0.6730, 0.477, -0.228, 0.0703, -0.0123]

		def lossSecnRb(U, nRb):
			return 10**(-3)*nRb*sigmaV*(self.pQDU(U, U_d, betaJ))

		popt, pcov = curve_fit(lossSecnRb, U, G, sigma = GErr, p0= [9], absolute_sigma=True, maxfev=10**6)

		Uplot = np.linspace(min(U), max(U), 100)

		return popt, np.sqrt(np.diag(pcov)), Uplot/self.kB*1000, lossSecnRb(Uplot, *popt)

	def fitUd(self, U, scaledG, scaledGErr):

		def fitFunc(U, Ud):
			#betaJ = [1, -0.6754, 0.4992, -0.228, 0.1165,  -0.0321, 0.00413] # in order of degree 1 to degree 6
			betaJ = [1, -0.6730, 0.477, -0.228, 0.0703, -0.0123, 0.0009]
			#betaJ = [1, -0.6730, 0.477, -0.228, 0.0703, -0.0123]
			UUd = U/Ud
			UUds = np.array([UUd**i for i in range(0,len(betaJ))])
			pqdu6 = np.dot(betaJ, UUds)
			return pqdu6


		popt, pcov = curve_fit(fitFunc, U, scaledG, sigma = scaledGErr, absolute_sigma=True, maxfev = 10**6, p0 = [2.5])

		Uplot = np.linspace(min(U), max(U), 100)

		return popt, np.sqrt(np.diag(pcov)), Uplot, fitFunc(Uplot, popt[0])

	def generateCorrections(self, Emaxes, T, Eoffset, sigmaVtot):
		def avE(x, a, T, j):
			# here a will end up being Emax - Emin (U - Eoffset)
			beta = 1/(self.kB*T)
			return 2 / np.sqrt(math.pi) * (beta)**(3/2) * x**(1/2) * (a-x)**(j) * np.exp(-beta*x)

		
		trapVal = []
		for Utrap in Emaxes:
			correction_U = []
			for j in range(5):
			# gather corrections for each trap depth 
				result, err = quad(avE, 0, Utrap - Eoffset, args = (Utrap - Eoffset, T, j+1))
				normalization, err2 = quad(avE, 0, Utrap - Eoffset, args = (Utrap - Eoffset, T, 0))
				correction_U.append(result / normalization)
			trapVal.append(np.array(correction_U))
		trapVal = np.array(trapVal)

		betaJ = [1, -0.6730, 0.477, -0.228, 0.0703, -0.0123, 0.0009]
		Ud = self.Ud(sigmaVtot)[0]

		trapVals = []
		for i in range(len(trapVal)):
			summed = [betaJ[j+1]*trapVal[i][j]/(Ud**(j+1)) for j in range(len(trapVal[i]))]
			trapVals.append(1 + sum(summed))
		trapVals = np.array(trapVals)

		return trapVals

	def fitLossSec_new(self, U, G, Gerr, T, Eoffset):

		# here U should be the maximum trap depth
		X = np.array([U, T, Eoffset], dtype = object)

		## figure out trap depth correction factors

		def avE(x, a, T, j):
			# here a will end up being Emax - Emin (U - Eoffset)
			beta = 1/(self.kB*T)
			return 2 / np.sqrt(math.pi) * (beta)**(3/2) * x**(1/2) * (a-x)**(j) * np.exp(-beta*x)
		def flatten(x):
			result = []
			for el in x:
				if hasattr(el, "__iter__"):
					result.extend(flatten(el))
				else:
					result.append(el)
			return result
		
		trapVal = []
		for i in range(len(U)):
			correction_U = []
			normalization, err2 = quad(avE, 0, U[i] - Eoffset[i], args = (U[i] - Eoffset[i], T[i], 0))
			for j in range(5):
			# gather corrections for each trap depth 
				result, err = quad(avE, 0, U[i] - Eoffset[i], args = (U[i] - Eoffset[i], T[i], j+1))
				correction_U.append(result / normalization)
			trapVal.append(np.array(correction_U))
		trapVal = np.array(trapVal)


		popt, pcov = curve_fit(self.lossSec_new, trapVal, G, sigma = Gerr, absolute_sigma = True, maxfev = 10**7, p0 = [6.45, 120])
		#Uplot = np.linspace(min(U), max(U), 100)

		# trapPlot = []
		# for Utrap in Uplot:
		#     correction_U = []
		#     for j in range(5):
		#     # gather corrections for each trap depth 
		#         result = quad(avE, 0, Utrap - Eoffset, args = (Utrap - Eoffset, T, j+1))[0]
		#         normalization = quad(avE, 0, Utrap - Eoffset, args = (Utrap - Eoffset, T, 0))[0]
		#         correction_U.append(result / normalization)
		#     trapPlot.append(np.array(correction_U))
		# trapPlot = np.array(trapPlot)

		sortedinputs = sorted(zip(trapVal, U), key = lambda x: x[1])
		sortedU = np.array([x[1] for x in sortedinputs])
		sortedT = np.array([x[0] for x in sortedinputs])
		#sortedU = np.linspace(min(U), max(U), 100)
		#sortedT = np.linspace(min(flatten(trapVal)), max(flatten(trapVal)),100)

		return popt, np.sqrt(np.diag(pcov)), sortedU/self.kB*1000, self.lossSec_new(sortedT, *popt)

	def lossSec_new(self, Useries, sigmaVtot, n_rb): # n_rb can be also 1/alpha, where R = alpha * nrb
		betaJ = [1, -0.6730, 0.477, -0.228, 0.0703, -0.0123, 0.0009]
		Ud = self.Ud(sigmaVtot)[0]

		## instead do this
		trapVal = []
		for i in range(len(Useries)):
			summed = [betaJ[j+1]*Useries[i][j]/(Ud**(j+1)) for j in range(len(Useries[i]))]
			trapVal.append(1 + sum(summed))
		trapVal = np.array(trapVal)
			
		return 10**(-3)*n_rb * sigmaVtot * (trapVal)


	def lossSec_new2(self, X, sigmaVtot, n_rb):
		Emaxes, T, Eoffset = X[0], X[1], X[2]       
		def MB(E, T, Emin):
				beta = 1/(self.kB * T)
				return 2 / np.sqrt(math.pi) * (E - Emin)**(1/2) * np.heaviside(E - Emin, 1) * beta**(3/2) * np.exp(-(E-Emin)*beta)

		betaJ = [1, -0.6730, 0.477, -0.228, 0.0703, -0.0123, 0.0009]

		def integrand(E, Emax, sigmaVtot, T, Eoffset):
			Ud = self.Ud(sigmaVtot)[0]
			u = Emax - E
			UUd = np.array([(u/Ud)**i * betaJ[i] for i in range(7)])
			return MB(E, T, Eoffset) * sum(UUd)

		values = []
		for i in range(len(Emaxes)):
			num, err = quad(integrand, Eoffset[i], Emaxes[i], args = (Emaxes[i], sigmaVtot, T[i], Eoffset[i]))
			den, err2 = quad(MB, Eoffset[i], Emaxes[i], args = (T[i], Eoffset[i]))
			values.append(num / den)

		return 10**(-3)*n_rb * sigmaVtot * np.array(values)


	def fitLossSec_new2(self, Emaxes, G, Gerr, T, Eoffset):
		Xinput = np.array([Emaxes, T, Eoffset], dtype=object)
		popt, pcov = curve_fit(self.lossSec_new2, Xinput, G, sigma = Gerr, p0 = [6.45, 136], absolute_sigma = True, maxfev=10**6) # [6.45, 136]
		#EmaxPlot = np.linspace(min(Emaxes), max(Emaxes), len(Emaxes))
		sortedXinput = sorted(zip(Emaxes, T, Eoffset), key = lambda x: x[0])
		Emaxesplot = np.array([x[0] for x in sortedXinput])
		Tplot = np.array([x[1] for x in sortedXinput])
		Eoffsetplot = np.array([x[2] for x in sortedXinput])
		Xplot = np.array([Emaxesplot, Tplot, Eoffsetplot])
		Yplot = self.lossSec_new2(Xplot, *popt)

		Yplot_upper = self.lossSec_new2(Xplot, popt[0] + np.sqrt(np.diag(pcov))[0], popt[1])
		Yplot_lower = self.lossSec_new2(Xplot, popt[0] - np.sqrt(np.diag(pcov))[0], popt[1])
		
		return popt, np.sqrt(np.diag(pcov)), Emaxesplot/self.kB*1000, Yplot, Yplot_upper, Yplot_lower


	def int_E_max_jth(self, Emax, j, Emin, T):
		# calculates integral of rho(E) * (E - Emax)^j from 0 to Emax
		summed = 0
		kT = self.kB*T
		umax = (Emax - Emin) / (kT)
		for k in range(j+1):
			pref = sc.binom(j,k) * (Emax)**(j - k) * (-1)**k
			for i in range(k+1):
				pref2 = sc.binom(k,i) * (Emin)**(k - i) * (kT)**i * sc.gamma(i + 3/2)*(sc.gammainc(i+3/2, umax))
				summed+= pref * (pref2)
		return summed * 2 / np.sqrt(math.pi)


	def analytical_pQDU(self, Emax, Ud, T, Emin):
		# calculates the integrated (normalized) loss rate 1 - pQDU(Emax/Ud)
		betaJ = [0.6730, -0.477, 0.228, -0.0703, 0.0123, -0.0009]
		
		summed = 0
		norm = self.int_E_max_jth(Emax, 0, Emin, T)
		for j in range(6):
			summed+= betaJ[j] * (1 / Ud)**(j+1) * self.int_E_max_jth(Emax, j+1, Emin, T)
		return 1-(summed / norm) 

	def lossSec_analytical(self, X, sigmaVtot, nRb):
		# function to fit and extract sigmaVtot, nRb (or alpha equivalently)
		# note that this is only for the distribution at t = 0 
		# since there is no integration over time here

		Emaxes, T, Emins = X[0], X[1], X[2]
		Ud = self.Ud(sigmaVtot)[0]
		return sigmaVtot * 10**(-3) * nRb * self.analytical_pQDU(Emaxes, Ud, T, Emins)

	def fitLossSec_analytical(self, Emaxes, G, Gerr, T, Eoffset, p0 = [6.45, 136]):
		# note that this fits for the non-heated case
		Xinput = np.array([Emaxes, T, Eoffset], dtype=object)
		popt, pcov = curve_fit(self.lossSec_analytical, Xinput, G, sigma = Gerr, p0 = p0, absolute_sigma = True, maxfev=10**6) # [6.45, 136]
		#EmaxPlot = np.linspace(min(Emaxes), max(Emaxes), len(Emaxes))
		sortedXinput = sorted(zip(Emaxes, T, Eoffset), key = lambda x: x[0])
		Emaxesplot = np.array([x[0] for x in sortedXinput])
		Tplot = np.array([x[1] for x in sortedXinput])
		Eoffsetplot = np.array([x[2] for x in sortedXinput])
		Xplot = np.array([Emaxesplot, Tplot, Eoffsetplot])
		Yplot = self.lossSec_analytical(Xplot, *popt)

		Yplot_upper = self.lossSec_analytical(Xplot, popt[0] + np.sqrt(np.diag(pcov))[0], popt[1])
		Yplot_lower = self.lossSec_analytical(Xplot, popt[0] - np.sqrt(np.diag(pcov))[0], popt[1])
		
		return popt, np.sqrt(np.diag(pcov)), Emaxesplot/self.kB*1000, Yplot, Yplot_upper, Yplot_lower

	def lossSec_analytical_normed(self, X, sigmaVtot):

		Emaxes, T, Emins = X[0], X[1], X[2]
		Ud = self.Ud(sigmaVtot)[0]
		return self.analytical_pQDU(Emaxes, Ud, T, Emins)

	def fitLossSec_analytical_normed(self, Emaxes, G, Gerr, T, Eoffset, p0 = [6.45]):

		Xinput = np.array([Emaxes, T, Eoffset], dtype=object)
		popt, pcov = curve_fit(self.lossSec_analytical_normed, Xinput, G, sigma = Gerr, p0 = p0, absolute_sigma = True, maxfev=10**6)
		#EmaxPlot = np.linspace(min(Emaxes), max(Emaxes), len(Emaxes))
		sortedXinput = sorted(zip(Emaxes, T, Eoffset), key = lambda x: x[0])
		Emaxesplot = np.array([x[0] for x in sortedXinput])
		Tplot = np.array([x[1] for x in sortedXinput])
		Eoffsetplot = np.array([x[2] for x in sortedXinput])
		Xplot = np.array([Emaxesplot, Tplot, Eoffsetplot])
		Yplot = self.lossSec_analytical_normed(Xplot, *popt)

		# Yplot_upper = self.lossSec_analytical(Xplot, popt[0] + np.sqrt(np.diag(pcov))[0], popt[1])
		# Yplot_lower = self.lossSec_analytical(Xplot, popt[0] - np.sqrt(np.diag(pcov))[0], popt[1])
		
		return popt, np.sqrt(np.diag(pcov)), Emaxesplot/self.kB*1000, Yplot#, Yplot_upper, Yplot_lower

	
	def int_E_cut_jth(self, Emax, j, T,Emin,Ecut):
		# calculates integral of rho(E) * (E - Emax)^j from 0 to Emax
		summed = 0
		kT = self.kB*T
		umax = np.minimum((Ecut - Emin) / (kT), (Emax - Emin) / (kT))
		for k in range(j+1):
			pref = sc.binom(j,k) * (Emax)**(j - k) * (-1)**k
			for i in range(k+1):
				pref2 = sc.binom(k,i) * (Emin)**(k - i) * (kT)**i * sc.gamma(i + 3/2)*(sc.gammainc(i+3/2, umax))
				summed+= pref * (pref2)
		return summed * 2 / np.sqrt(math.pi)
	
	def analytical_pQDU_Ecut(self, Emax, Ud, T, Emin, Ecut):
		# calculates the integrated (normalized) loss rate 1 - pQDU(Emax/Ud)
		betaJ = [0.6730, -0.477, 0.228, -0.0703, 0.0123, -0.0009]
		
		summed = 0
		norm = self.int_E_cut_jth(Emax, 0, T, Emin, Ecut)
		for j in range(6):
			summed+= betaJ[j] * (1 / Ud)**(j+1) * self.int_E_cut_jth(Emax, j+1, T, Emin, Ecut)
		return 1-(summed / norm) 
		
		
	
	def lossSec_analytical_precut(self, X, sigmaVtot, nRb):
		# function to fit and extract sigmaVtot, nRb (or alpha equivalently)
		# note that this is only for the distribution at t = 0 
		# since there is no integration over time here

		Emaxes, T, Emins, Ecuts = X[0], X[1], X[2], X[3]
		Ud = self.Ud(sigmaVtot)[0]
		return sigmaVtot * 10**(-3) * nRb * self.analytical_pQDU_Ecut(Emaxes, Ud, T, Emins, Ecuts)
	
	def fitLossSec_analytical_precut(self, Emaxes, G, Gerr, T, Eoffset, Ecuts, p0 = [6.45, 136]):
		# note that this fits for the non-heated case
		Xinput = np.array([Emaxes, T, Eoffset, Ecuts], dtype=object)
		popt, pcov = curve_fit(self.lossSec_analytical_precut, Xinput, G, sigma = Gerr, p0 = p0, absolute_sigma = True, maxfev=10**6) # [6.45, 136]
		#EmaxPlot = np.linspace(min(Emaxes), max(Emaxes), len(Emaxes))
		sortedXinput = sorted(zip(Emaxes, T, Eoffset, Ecuts), key = lambda x: x[0])
		Emaxesplot = np.array([x[0] for x in sortedXinput])
		Tplot = np.array([x[1] for x in sortedXinput])
		Eoffsetplot = np.array([x[2] for x in sortedXinput])
		Ecutplot = np.array([x[3] for x in sortedXinput])
		Xplot = np.array([Emaxesplot, Tplot, Eoffsetplot, Ecutplot])
		Yplot = self.lossSec_analytical_precut(Xplot, *popt)

		Yplot_upper = self.lossSec_analytical_precut(Xplot, popt[0] + np.sqrt(np.diag(pcov))[0], popt[1])
		Yplot_lower = self.lossSec_analytical_precut(Xplot, popt[0] - np.sqrt(np.diag(pcov))[0], popt[1])
		
		return popt, np.sqrt(np.diag(pcov)), Emaxesplot/self.kB*1000, Yplot, Yplot_upper, Yplot_lower
	
	def int_E_cut_jth_m_t(self, Emax, j, T,Emin,Ecut, m):
		summed = 0
		
		# here, the CDF is defined such that 
		# it hits 1 at E=Ecut, and has a slope of m
		# (relative to this) above that point until
		# Emax
		
		def MBEDist(E, T,C, B=1):
			# Energy is always positive (this works better than just setting bounds for curve_fit)
			E = E - C
			ind_neg = np.where(E < 0)[0]
			E[ind_neg] = 0
			
			# max trap depth - CDF levels off there

			# Temperature is always positive (same reason as above)
			T = abs(T)
			
			t1 = erf(np.sqrt(E / (self.kB*T)))
			t2 = 2/np.sqrt(math.pi)*np.exp(-E/(self.kB*T)) * np.sqrt(E / (self.kB*T))
			
			t1max = erf(np.sqrt(Emax / (self.kB*T)))
			t2max = 2/np.sqrt(math.pi)*np.exp(-Emax/(self.kB*T)) * np.sqrt(Emax / (self.kB*T))
			
			ValMax = B*(t1max - t2max)
	
			Values = B*(t1 - t2)
						
			return Values
		
		kT = self.kB*T
		umax = np.minimum((Ecut - Emin) / (kT), (Emax - Emin) / (kT))
		for k in range(j+1):
			pref = sc.binom(j,k) * (Emax)**(j - k) * (-1)**k
			for i in range(k+1):
				pref2 = sc.binom(k,i) * (Emin)**(k - i) * (kT)**i * sc.gamma(i + 3/2)*(sc.gammainc(i+3/2, umax))
				summed+= pref * (pref2)
				
		summed *= 2 / np.sqrt(math.pi)
		# normalize the pdf such that the CDF would reach 1 at E = Ecut
		summed /= MBEDist(np.array([Ecut]), T, Emin)[0]
				
		# now add in the terms corresponding to those above the precut
		for l in range(j+1):
			summed += np.heaviside(Emax - Ecut, 1) * m/(self.kB) * 1000 * sc.binom(j,l) * (-1)**(l) * (Emax)**(j-l) / (l+1) * (Emax**(l+1) - Ecut**(l+1))
			#summed += MBEDist(Ecut, T, Emin) * sc.binom(j,l) * (-1)**(l) * (Emax)**(j-l) / (l+1) * (Emax**(l+1) - Ecut**(l+1))
			
		try:
			len(umax)
			inds = np.where(umax < 0)[0]
			summed[inds] = 0
		except:
			if umax <0:
				summed = 0
		
		return summed
	
	def analytical_pQDU_Ecut_m_t(self, Emax, Ud, T, Emin, Ecut, m):
		# calculates the integrated (normalized) loss rate 1 - pQDU(Emax/Ud)
		betaJ = [0.6730, -0.477, 0.228, -0.0703, 0.0123, -0.0009]
		
		summed = 0
		norm = self.int_E_cut_jth_m_t(Emax, 0, T, Emin, Ecut, m)
		for j in range(6):
			summed+= betaJ[j] * (1 / Ud)**(j+1) * self.int_E_cut_jth_m_t(Emax, j+1, T, Emin, Ecut, m)
		return 1-(summed / norm) 
	
	def lossSec_analytical_precut_m_t(self, X, sigmaVtot, nRb):
		# function to fit and extract sigmaVtot, nRb (or alpha equivalently)
		# note that this is only for the distribution at t = 0 
		# since there is no integration over time here

		Emaxes, T, Emins, Ecuts, ms = X[0], X[1], X[2], X[3]
		Ud = self.Ud(sigmaVtot)[0]
		return sigmaVtot * 10**(-3) * nRb * self.analytical_pQDU_Ecut_m_t(Emaxes, Ud, T, Emins, Ecuts, ms)
	
	def fitLossSec_analytical_precut_m_t(self, Emaxes, G, Gerr, T, Eoffset, Ecuts, ms, p0 = [6.45, 70]):
		# note that this fits for the non-heated case
		Xinput = np.array([Emaxes, T, Eoffset, Ecuts, ms], dtype=object)
		popt, pcov = curve_fit(self.lossSec_analytical_precut_m_t, Xinput, G, sigma = Gerr, p0 = p0, absolute_sigma = True, maxfev=10**6) # [6.45, 136]
		#EmaxPlot = np.linspace(min(Emaxes), max(Emaxes), len(Emaxes))
		sortedXinput = sorted(zip(Emaxes, T, Eoffset, Ecuts, ms), key = lambda x: x[0])
		Emaxesplot = np.array([x[0] for x in sortedXinput])
		Tplot = np.array([x[1] for x in sortedXinput])
		Eoffsetplot = np.array([x[2] for x in sortedXinput])
		Ecutplot = np.array([x[3] for x in sortedXinput])
		msplot = np.array([x[4] for x in sortedXinput])
		Xplot = np.array([Emaxesplot, Tplot, Eoffsetplot, Ecutplot, msplot])
		Yplot = self.lossSec_analytical_precut_m_t(Xplot, *popt)

		Yplot_upper = self.lossSec_analytical_precut_m_t(Xplot, popt[0] + np.sqrt(np.diag(pcov))[0], popt[1])
		Yplot_lower = self.lossSec_analytical_precut_m_t(Xplot, popt[0] - np.sqrt(np.diag(pcov))[0], popt[1])
		
		return popt, np.sqrt(np.diag(pcov)), Emaxesplot/self.kB*1000, Yplot, Yplot_upper, Yplot_lower
	
	
		
	
	
			
			

		





		


	



		
			





