from os import environ
import MeCab

environ["MECABRC"] = "/etc/mecabrc"
tagger = MeCab.Tagger("-Owakati")
cou = 0
in_str = "<o>お肉は</o><r>柔らかく</r>市販のたれとは違いさすがプロの味付けに満足ですね。<e>美味しくいただきました。</e>"
nin_str = in_str.replace('<o>','')
nin_str = nin_str.replace('</o>','')
nin_str = nin_str.replace('<r>','')
nin_str = nin_str.replace('</r>','')
nin_str = nin_str.replace('<e>','')
nin_str = nin_str.replace('</e>','')
#print(in_str)
#print(nin_str)
words = tagger.parse(nin_str).split()
#nwords = str(words)
#nwords = nwords.replace(',','\n')
#nwords = nwords.replace("'",'')
#nwords = nwords.replace("[",'')
#nwords = nwords.replace("]",'')
#nwords = nwords.strip()
#print(words)

for val in words:
    cou += len(val)
    print(val,cou)

