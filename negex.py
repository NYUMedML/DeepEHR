import re

def sortRules (ruleList):
    """Return sorted list of rules.
    
    Rules should be in a tab-delimited format: 'rule\t\t[four letter negation tag]'
    Sorts list of rules descending based on length of the rule, 
    splits each rule into components, converts pattern to regular expression,
    and appends it to the end of the rule. """
    ruleList.sort(key = len, reverse = True)
    sortedList = []
    for rule in ruleList:
        s = rule.strip()
        s= s.split('\t')

        splitTrig = s[0].split()
        trig = r'\s+'.join(splitTrig)
        pattern = r'\b(' + trig + r')\b'
        s.append(re.compile(pattern, re.IGNORECASE))
        sortedList.append(s)
    return sortedList

class negTagger(object):
    '''Take a sentence and tag negation terms and negated phrases.
    
    Keyword arguments:
    sentence -- string to be tagged
    phrases  -- list of phrases to check for negation
    rules    -- list of negation trigger terms from the sortRules function
    negP     -- tag 'possible' terms as well (default = True)    '''
    def __init__(self, sentence = '', phrases = None, rules = None, 
                 negP = True):
        self.__sentence = sentence
        self.__phrases = phrases
        self.__rules = rules
        self.__negTaggedSentence = ''
        self.__scopesToReturn = []
        self.__negationFlag = None
        
        filler = '_'
        
        for rule in self.__rules:
            reformatRule = re.sub(r'\s+', filler, rule[0].strip())
            self.__sentence = rule[3].sub (' ' + rule[2].strip()
                                                          + reformatRule
                                                          + rule[2].strip() + ' ', self.__sentence)
        for phrase in self.__phrases:
            phrase = re.sub(r'([.^$*+?{\\|()[\]])', r'\\\1', phrase)
            splitPhrase = phrase.split()
            joiner = r'\W+'
            joinedPattern = r'\b' + joiner.join(splitPhrase) +  r'\b'
            reP = re.compile(joinedPattern, re.IGNORECASE)
            m = reP.search(self.__sentence)
            if m:
                self.__sentence = self.__sentence.replace(m.group(0), '[PHRASE]'
                                                          + re.sub(r'\s+', filler, m.group(0).strip())
                                                          + '[PHRASE]')
    
#        Exchanges the [PHRASE] ... [PHRASE] tags for [NEGATED] ... [NEGATED] 
#        based on PREN, POST rules and if negPoss is set to True then based on 
#        PREP and POSP, as well.
#        Because PRENEGATION [PREN} is checked first it takes precedent over
#        POSTNEGATION [POST]. Similarly POSTNEGATION [POST] takes precedent over
#        POSSIBLE PRENEGATION [PREP] and [PREP] takes precedent over POSSIBLE 
#        POSTNEGATION [POSP].
              
        overlapFlag = 0
        prenFlag = 0
        postFlag = 0
        prePossibleFlag = 0
        postPossibleFlag = 0
        
        sentenceTokens = self.__sentence.split()
        sentencePortion = ''
        aScopes = []
        sb = []
        #check for [PREN]
        for i in range(len(sentenceTokens)):
            if sentenceTokens[i][:6] == '[PREN]':
                prenFlag = 1
                overlapFlag = 0

            if sentenceTokens[i][:6] in ['[CONJ]', '[PSEU]', '[POST]', '[PREP]', '[POSP]']:
                overlapFlag = 1
            
            if i+1 < len(sentenceTokens):
                if sentenceTokens[i+1][:6] == '[PREN]':
                    overlapFlag = 1
                    if sentencePortion.strip():
                        aScopes.append(sentencePortion.strip())
                    sentencePortion = ''
            
            if prenFlag == 1 and overlapFlag == 0:
                sentenceTokens[i] = sentenceTokens[i].replace('[PHRASE]', '[NEGATED]')
                sentencePortion = sentencePortion + ' ' + sentenceTokens[i]
            
            sb.append(sentenceTokens[i])
        
        if sentencePortion.strip():
            aScopes.append(sentencePortion.strip())
        
        sentencePortion = ''
        sb.reverse()
        sentenceTokens = sb
        sb2 = []
        # Check for [POST]
        for i in range(len(sentenceTokens)):
            if sentenceTokens[i][:6] == '[POST]':
                postFlag = 1
                overlapFlag = 0

            if sentenceTokens[i][:6] in ['[CONJ]', '[PSEU]', '[PREN]', '[PREP]', '[POSP]']:
                overlapFlag = 1
            
            if i+1 < len(sentenceTokens):
                if sentenceTokens[i+1][:6] == '[POST]':
                    overlapFlag = 1
                    if sentencePortion.strip():
                        aScopes.append(sentencePortion.strip())
                    sentencePortion = ''
            
            if postFlag == 1 and overlapFlag == 0:
                sentenceTokens[i] = sentenceTokens[i].replace('[PHRASE]', '[NEGATED]')
                sentencePortion = sentenceTokens[i] + ' ' + sentencePortion
            
            sb2.insert(0, sentenceTokens[i])
        
        if sentencePortion.strip():
            aScopes.append(sentencePortion.strip())
        
        sentencePortion = ''
        self.__negTaggedSentence = ' '.join(sb2)
        
        if negP:
            sentenceTokens = sb2
            sb3 = []
            # Check for [PREP]
            for i in range(len(sentenceTokens)):
                if sentenceTokens[i][:6] == '[PREP]':
                    prePossibleFlag = 1
                    overlapFlag = 0

                if sentenceTokens[i][:6] in ['[CONJ]', '[PSEU]', '[POST]', '[PREN]', '[POSP]']:
                    overlapFlag = 1
            
                if i+1 < len(sentenceTokens):
                    if sentenceTokens[i+1][:6] == '[PREP]':
                        overlapFlag = 1
                        if sentencePortion.strip():
                            aScopes.append(sentencePortion.strip())
                        sentencePortion = ''
            
                if prePossibleFlag == 1 and overlapFlag == 0:
                    sentenceTokens[i] = sentenceTokens[i].replace('[PHRASE]', '[POSSIBLE]')
                    sentencePortion = sentencePortion + ' ' + sentenceTokens[i]
            
                sb3 = sb3 + ' ' + sentenceTokens[i]
        
            if sentencePortion.strip():
                aScopes.append(sentencePortion.strip())
            
            sentencePortion = ''
            sb3.reverse()
            sentenceTokens = sb3 
            sb4 = []
            # Check for [POSP]
            for i in range(len(sentenceTokens)):
                if sentenceTokens[i][:6] == '[POSP]':
                    postPossibleFlag = 1
                    overlapFlag = 0

                if sentenceTokens[i][:6] in ['[CONJ]', '[PSEU]', '[PREN]', '[PREP]', '[POST]']:
                    overlapFlag = 1
            
                if i+1 < len(sentenceTokens):
                    if sentenceTokens[i+1][:6] == '[POSP]':
                        overlapFlag = 1
                        if sentencePortion.strip():
                            aScopes.append(sentencePortion.strip())
                        sentencePortion = ''
            
                if postPossibleFlag == 1 and overlapFlag == 0:
                    sentenceTokens[i] = sentenceTokens[i].replace('[PHRASE]', '[POSSIBLE]')
                    sentencePortion = sentenceTokens[i] + ' ' + sentencePortion
            
                sb4.insert(0, sentenceTokens[i])
        
            if sentencePortion.strip():
                aScopes.append(sentencePortion.strip())
            
            self.__negTaggedSentence = ' '.join(sb4)
            
        if '[NEGATED]' in self.__negTaggedSentence:
            self.__negationFlag = 'negated'
        elif '[POSSIBLE]' in self.__negTaggedSentence:
            self.__negationFlag = 'possible'
        else:
            self.__negationFlag = 'affirmed'
        
        self.__negTaggedSentence = self.__negTaggedSentence.replace(filler, ' ')
        
        for line in aScopes:
            tokensToReturn = []
            thisLineTokens = line.split()
            for token in thisLineTokens:
                if token[:6] not in ['[PREN]', '[PREP]', '[POST]', '[POSP]']:
                    tokensToReturn.append(token)
            self.__scopesToReturn.append(' '.join(tokensToReturn))

    def getNegTaggedSentence(self):
        return self.__negTaggedSentence
    def getNegationFlag(self):
        return self.__negationFlag
    def getScopes(self):
        return self.__scopesToReturn
    
    def __str__(self):
        text = self.__negTaggedSentence
        text += '\t' + self.__negationFlag
        text += '\t' + '\t'.join(self.__scopesToReturn)
