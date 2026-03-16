import glob,csv
import pandas as pd

def suoyin_1(dizhi_1):
    suoyin_tt = {}
    for ii in dizhi_1:
        if ii not in suoyin_tt:
            suoyin_tt[ii] = 0
        sheet = pd.read_excel(ii)
        shuhh = list(sheet.keys())
        suoyin_tt[ii] = shuhh[1:]
    return suoyin_tt

def suoyin_2(dizhi_2):
    suoyin_tt_2 = {}
    
    ex_data = pd.read_excel(dizhi_2[0]) #默认读取第一个sheet的内容
    
    head_list = list(ex_data.columns) #拿到表头: [A, B, C, D]
    # print(head_list)
    for ii in range(len(head_list)):
        if head_list[ii] not in suoyin_tt_2:
            suoyin_tt_2[head_list[ii]] = []
        suoyin_tt_2[head_list[ii]] = list(ex_data.iloc[:,ii])
    return suoyin_tt_2
if __name__ == "__main__":
    
    
    
    
    dizhi_1 = glob.glob("data\\*")
    suoyin_tt = suoyin_1(dizhi_1)
    
    dizhi_2 = glob.glob("data2\\*")
    suoyin_tt_2 = suoyin_2(dizhi_2)
    
    dizhi_3 = r"data2\\53yinzi.xlsx"
    suyin_data = pd.read_excel(dizhi_3)
    head_list = list(suyin_data.columns)
    sss = list(suyin_data["btc_daily_price"])#.iloc[:,2])
    with open("..\\data\\suoyin.csv","a",newline='',encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([str(qqq) for qqq in list(suoyin_tt_2["日期"])])
        writer.writerow(sss)
    
    
    for ii in suoyin_tt.keys():
        
        ss = ii.strip().split("\\")[-1].split(".")[0]
        with open("canren\\%s_shuju.csv"%ss,"a",newline='',encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([str(qqq) for qqq in list(suoyin_tt_2["日期"])])
        for qq in suoyin_tt[ii]:
            aaa = suoyin_tt_2[qq]
        
            with open("canren\\%s_shuju.csv"%ss,"a",newline='',encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(aaa)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    