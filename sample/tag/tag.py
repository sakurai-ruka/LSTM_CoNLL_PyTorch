from os import environ
import MeCab

environ["MECABRC"] = "/etc/mecabrc"
tagger = MeCab.Tagger("-Owakati")

in_str = "<o>お肉は</o><r>柔らかく</r>市販のたれとは違いさすがプロの味付けに満足ですね。<e>美味しくいただきました。</e>"
nin_str = in_str.replace('<o>','')
nin_str = nin_str.replace('</o>','')
nin_str = nin_str.replace('<r>','')
nin_str = nin_str.replace('</r>','')
nin_str = nin_str.replace('<e>','')
nin_str = nin_str.replace('</e>','')
words = tagger.parse(nin_str).split()
#words = str(words)
def flat_and_extract_tag_pos(in_str):
    out_str = ""
    nout_str = ""
    cou = 0
    info_dict = {}
    while True:
        # 開始タグの取得
        spos = in_str.find("<")
        # import ipdb; ipdb.set_trace()
        out_spos = spos + len(out_str)
        if spos == -1:
            out_str += in_str
            #nout_str = out_str
            break
        elif spos > 0:
            out_str += in_str[:spos]
        nout_str = out_str
        #nout_str = nout_str.replace('\n','')
        print(nout_str)
        cou += len(nout_str)
        #print(cou)
        ncou = cou - len(nout_str)+1
        #print(ncou)
        #print(len(nout_str))
        tag_name = in_str[spos+1:spos+2]
        in_str = in_str[spos+3:]
        # 間の文字列を取得
        epos = in_str.find("<")
        out_epos = epos + len(out_str)
        if epos < -1:
            raise RuntimeError("cannot find tag end")
        elif epos == 0:
            raise RuntimeError("empty content")
        #print(in_str[:epos])
        out_str = in_str[:epos]
        #out_str = in_str[:epos]
        tag_name_end = in_str[epos+2:epos+3]
        if tag_name != tag_name_end:
            raise RuntimeError("tag name mismatch")
        in_str = in_str[epos+4:]
        info_dict[tag_name] = [out_spos, out_epos]
        print(out_str,tag_name,sep="\n")
        #print(len(out_str))
        #文字数のカウント
        cou +=len(out_str)
        #次の区間の始まりの文字数
        ncou = cou-len(out_str)+1
        #print(cou)
        #print(ncou)
        if tag_name == "o":
            in_o = ncou
            out_o = cou
            print("oは"+str(in_o)+"-"+str(out_o)+"文字目")
        elif tag_name =="r":
            in_r = ncou
            out_r = cou
            print("rは"+str(in_r)+"-"+str(out_r)+"文字目")
        elif tag_name =="e":
            in_e = ncou
            out_e = cou
            print("eは"+str(in_r)+"-"+str(out_r)+"文字目")
        #print("eはn-n番目")
        out_str = ""
    return out_str,info_dict


flatten_str, tag_pos = flat_and_extract_tag_pos(in_str)
