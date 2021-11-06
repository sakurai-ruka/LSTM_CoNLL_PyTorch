from os import environ
import MeCab
import pickle

environ["MECABRC"] = "/etc/mecabrc"
tagger = MeCab.Tagger("-Owakati")
#in_str = "<o>お肉は</o><r>柔らかく</r>市販のたれとは違いさすがプロの味付けに満足ですね。<e>美味しくいただきました。</e>"
# in_str = "<o>012</o>3456<e>789</e>"
#file =open('test.txt','r')
#in_str =file.read()
#file.close()
# "お肉は柔らかく市販のたれとは違いさすがプロの味付けに満足ですね。美味しくいただきました。"
# o -> 0, 3
# r -> 3, 7
# e -> 32, 44
# (1) タグを取り除き，取り除いた位置を記録する
def flat_and_extract_tag_pos(in_str):
    out_str = ""
    info_dict = {}
    while True:
        # 開始タグの取得
        spos = in_str.find("<")
        # import ipdb; ipdb.set_trace()
        out_spos = spos + len(out_str)
        if spos == -1:
            out_str += in_str
            break
        elif spos > 0:
            out_str += in_str[:spos]
        tag_name = in_str[spos+1:spos+2]
        in_str = in_str[spos+3:]
        # 間の文字列を取得
        epos = in_str.find("<")
        out_epos = epos + len(out_str)
        if epos < -1:
            raise RuntimeError("cannot find tag end")
        elif epos == 0:
            raise RuntimeError("empty content")
        out_str += in_str[:epos]
        tag_name_end = in_str[epos+2:epos+3]
        if tag_name != tag_name_end:
            raise RuntimeError("tag name mismatch")
        in_str = in_str[epos+4:]
        #
        info_dict[tag_name] = [out_spos, out_epos]
    return out_str, info_dict

def get_wakachi(in_str):
    nin_str = in_str.replace('<o>','')
    nin_str = nin_str.replace('</o>','')
    nin_str = nin_str.replace('<r>','')
    nin_str = nin_str.replace('</r>','')
    nin_str = nin_str.replace('<e>','')
    nin_str = nin_str.replace('</e>','')
    words = tagger.parse(nin_str).split()

    cou = 0
    counts = []
    for word in words:
        counts.append((cou, cou + len(word)))
        cou += len(word)
    return words, counts

def get_label(tag_pos, count):
    st, en = count
    for tag, pos in tag_pos.items():
        if tag == 'o':
            tag = 't'
        p_st, p_en = pos
        if (st <= p_st and en > p_st) or \
           (st < p_en and en > p_st):
            return tag
    return 'o'

def get_word_label(words, counts, tag_pos):
    results = []
    for word, count in zip(words, counts):
        label = get_label(tag_pos, count)
        results.append((word,'*','*' ,label))
        #results.append(('\n'))
    return results
    
def load_dumps(f):
    obj = []
    while 1:
        try:
            obj.extend(pickle.load(f))
        except:
            break
        return obj

def generate_word_label_seq(in_str):
    words, counts = get_wakachi(in_str)
    flatten_str, tag_pos = flat_and_extract_tag_pos(in_str)
    #global results
    #for word, count in zip(words, counts):
       # label = get_label(tag_pos, count)
        #results = (word, '*','*',label)
        #results = str(results)
        #word_label_seq = results
    word_label_seq = get_word_label(words, counts, tag_pos)
        #with open('kakikomi.pkl','ab') as f:
           # pickle.dump(results,f)
    file1 = open('kakikomi.txt','a')
    #file1.write(results)
    file1.write(str(word_label_seq))
    file1.write('\n')
    file1.close()
    #print(word_label_seq)
    return word_label_seq
    #return word_label_seq
def exclude(in_str):
    in_str.strip("'""("")")
    file1 = open('kakikomi2.txt','a')
    file1.write(in_str)
    file1.write('\n')
    file1.close()

file2 = open('niku1105.txt','r')
while True:
    in_str = file2.readline()
    if in_str:
        generate_word_label_seq(in_str)
        #with open('kakikomi.pkl',mode='wb') as f1:
           # pickle.dump(word_label_seq,f1)
        #print(word_label_seq)
    else:
        break
file2.close()
#ile3 = open('kakikomi.txt','r')
#while True:
    #in_str =file3.readline()
    #if in_str:
        #exclude(in_str)
    #else:
        #break
#file3 = open('kakikomi.txt','r')
#while True:
    #in_str =file3.readline()
    #if in_str:
       #with open('kakikomi.pkl','ab') as f:
           #pickle.dump(in_str,f)
    #else:
        #break
#file3.close()
