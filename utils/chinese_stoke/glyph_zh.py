def initDict(path):
   dict = {}
   with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            splits = line.strip('\n').split(' ')
            key = splits[0]
            value = splits[1]
            dict[key] = value
   return dict
   
import os

current_directory = os.path.dirname(os.path.abspath(__file__))

bihuashuDict = initDict(current_directory+'/db/bihuashu_2w.txt')
hanzijiegouDict = initDict(current_directory+'/db/hanzijiegou_2w.txt')
pianpangbushouDict = initDict(current_directory+'/db/pianpangbushou_2w.txt')
sijiaobianmaDict = initDict(current_directory+'/db/sijiaobianma_2w.txt')

hanzijiegouRate = 10
sijiaobianmaRate = 8
pianpangbushouRate = 6
bihuashuRate = 2


def bihuashuSimilar(charOne, charTwo): 
    if charOne not in bihuashuDict or charTwo not in bihuashuDict:
        return 0.0
    valueOne = bihuashuDict[charOne]
    valueTwo = bihuashuDict[charTwo]
    
    numOne = int(valueOne)
    numTwo = int(valueTwo)
    
    diffVal = 1 - abs((numOne - numTwo) / max(numOne, numTwo))
    return bihuashuRate * diffVal * 1.0

    
def hanzijiegouSimilar(charOne, charTwo): 
    if charOne not in hanzijiegouDict or charTwo not in hanzijiegouDict:
        return 0.0
    valueOne = hanzijiegouDict[charOne]
    valueTwo = hanzijiegouDict[charTwo]
    
    if valueOne == valueTwo:
        return hanzijiegouRate * 1
    return 0


def sijiaobianmaSimilar(charOne, charTwo): 
    if charOne not in sijiaobianmaDict or charTwo not in sijiaobianmaDict:
        return 0.0
    valueOne = sijiaobianmaDict[charOne]
    valueTwo = sijiaobianmaDict[charTwo]
    
    totalScore = 0.0
    minLen = min(len(valueOne), len(valueTwo))
    
    for i in range(minLen):
        if valueOne[i] == valueTwo[i]:
            totalScore += 1.0
    
    totalScore = totalScore / minLen * 1.0
    return totalScore * sijiaobianmaRate


def pianpangbushoutSimilar(charOne, charTwo): 
    if charOne not in pianpangbushouDict or charTwo not in pianpangbushouDict:
        return 0.0
    valueOne = pianpangbushouDict[charOne]
    valueTwo = pianpangbushouDict[charTwo]
    
    if valueOne == valueTwo:
        return pianpangbushouRate * 1
    return 0 
    

def glyph_similar(charOne, charTwo):
    if charOne == charTwo:
        return 1.0
    
    sijiaoScore = sijiaobianmaSimilar(charOne, charTwo)  
    jiegouScore = hanzijiegouSimilar(charOne, charTwo)
    bushouScore = pianpangbushoutSimilar(charOne, charTwo)
    bihuashuScore = bihuashuSimilar(charOne, charTwo)
    
    totalScore = sijiaoScore + jiegouScore + bushouScore + bihuashuScore;    
    totalRate = hanzijiegouRate + sijiaobianmaRate + pianpangbushouRate + bihuashuRate
    
    result = totalScore*1.0 / totalRate * 1.0

    return result
