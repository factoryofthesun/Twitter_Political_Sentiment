import pandas as pd
import stanza

depparser = stanza.Pipeline("en", processors="tokenize,mwt,pos,lemma,depparse", use_gpu = True)
test = depparser("this is a test. I am the biggest fat. Listen here bucko! :)")
print(test)
print(test.sentences[0])
