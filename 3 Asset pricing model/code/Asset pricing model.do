############################## Data processing #################################
use "fivefactor_daily.xlsx.dta", clear
rename trddy tradingdate
format tradingdate %td
destring mkt_rf smb hml umd rmw cma, replace force
save "factors.dta", replace
* currency
use "us_dollar_index2024.xlsx.dta", clear
rename 涨跌幅 US
drop 交易量
save "us_dollar_index.dta", replace
use "usd_aud2024.xlsx.dta", clear
rename 涨跌幅 AUD
drop 交易量
save "usd_aud.dta", replace
use "usd_cad2024.xlsx.dta", clear
rename 涨跌幅 CAD
drop 交易量
save "usd_cad.dta", replace
use "usd_eur2024.xlsx.dta", clear
rename 涨跌幅 EUR
drop 交易量
save "usd_eur.dta", replace
use "usd_gbp2024.xlsx.dta", clear
rename 涨跌幅 GBP
drop 交易量
save "usd_gbp.dta", replace
use "usd_jpy2024.xlsx.dta", clear
rename 涨跌幅 JPY
drop 交易量
save "usd_jpy.dta", replace
use "usd_dkk2024.xlsx.dta", clear
rename 涨跌幅 DKK
drop 交易量
save "usd_dkk.dta", replace
use "us_dollar_index.dta", clear
mergemany 1:1 us_dollar_index usd_aud usd_cad usd_eur usd_gbp usd_jpy usd_dkk, match(日期)
rename 日期 tradingdate
keep tradingdate US AUD CAD EUR GBP JPY DKK
save "currency.dta", replace
* commodity
use "brent_crude_futures.dta", clear
rename 涨跌幅 brent
save "brent_crude_futures.dta", replace
use "gold_futures.dta", clear
rename 涨跌幅 gold
save "gold_futures.dta", replace
use "copper_futures.dta", clear
rename 涨跌幅 copper
save "copper_futures.dta", replace
mergemany 1:1 brent_crude_futures gold_futures copper_futures, match(日期)
rename 日期 tradingdate
keep tradingdate brent gold copper
save "commodity.dta", replace
* Cryptocurrency
use "btc.dta", clear
gen date = substr(snapped_at,1,10)
rename market_cap btc_mc
gen btc_return = ln(price[_n]/price[_n-1]) 
keep date btc_return btc_mc
save "btc.dta", replace
use "btcprice", clear
gen date = substr(snapped_at,1,10)
rename price btc_price
keep date btc_price
save "btcprice.dta", replace
use "itc.dta", clear
gen date = substr(snapped_at,1,10)
rename market_cap itc_mc
gen itc_return = ln(price[_n]/price[_n-1]) 
keep date itc_return itc_mc
save "itc.dta", replace
use "eth.dta", clear
gen date = substr(snapped_at,1,10)
rename market_cap eth_mc
gen eth_return = ln(price[_n]/price[_n-1]) 
keep date eth_return eth_mc
save "eth.dta", replace
use "dash.dta", clear
gen date = substr(snapped_at,1,10)
rename market_cap dash_mc
gen dash_return = ln(price[_n]/price[_n-1]) 
keep date dash_return dash_mc
save "dash.dta", replace



############################## Data integration ################################
mergemany 1:1 btc dash eth itc btcprice, match(date)
rename itc_mc ltc_mc 
rename itc_return ltc_return
sort date
gen tradingdate = date(date,"YMD")
format tradingdate %td
gen year = year(tradingdate)
gen quarter = quarter(tradingdate)
gen month = month(tradingdate)
gen dow = dow(tradingdate)
order tradingdate year quarter month dow
tsset tradingdate
drop date 
drop in 1/853
drop in 3319/3374
merge m:m tradingdate using "factors.dta"
drop if _merge==2
drop _merge
merge m:m tradingdate using "currency.dta"
drop if _merge==2
drop _merge
merge m:m tradingdate using "commodity.dta"
drop if _merge==2
drop _merge
gen btc_r_positive_total = 1 if btc_return>0 & btc_return!=.
gen dash_r_positive_total = 1 if dash_return>0 & dash_return!=.
gen eth_r_positive_total = 1 if eth_return>0 & eth_return!=.
gen ltc_r_positive_total = 1 if ltc_return>0 & ltc_return!=.
egen btc_r_positive = total(btc_r_positive_total)
egen dash_r_positive = total(dash_r_positive_total)
egen eth_r_positive = total(eth_r_positive_total)
egen ltc_r_positive = total(ltc_r_positive_total)
egen sd_btc = sd(btc_return)
egen mean_btc = mean(btc_return)
gen btc_sharperatio = sd_btc/mean_btc
egen sd_dash = sd(dash_return)
egen mean_dash = mean(dash_return)
gen dash_sharperatio = sd_dash/mean_dash
egen sd_eth = sd(eth_return)
egen mean_eth = mean(eth_return)
gen eth_sharperatio = sd_eth/mean_eth
egen sd_ltc = sd(ltc_return)
egen mean_ltc = mean(ltc_return)
gen ltc_sharperatio = sd_ltc/mean_ltc
gen btc_dis_5 = 1 if btc_return<-0.05
gen btc_dis_10 = 1 if btc_return<-0.1
gen btc_dis_20 = 1 if btc_return<-0.2
gen btc_dis_30 = 1 if btc_return<-0.3
gen dash_dis_5 = 1 if dash_return<-0.05
gen dash_dis_10 = 1 if dash_return<-0.1
gen dash_dis_20 = 1 if dash_return<-0.2
gen dash_dis_30 = 1 if dash_return<-0.3
gen eth_dis_5 = 1 if eth_return<-0.05
gen eth_dis_10 = 1 if eth_return<-0.1
gen eth_dis_20 = 1 if eth_return<-0.2
gen eth_dis_30 = 1 if eth_return<-0.3
gen ltc_dis_5 = 1 if ltc_return<-0.05
gen ltc_dis_10 = 1 if ltc_return<-0.1
gen ltc_dis_20 = 1 if ltc_return<-0.2
gen ltc_dis_30 = 1 if ltc_return<-0.3
gen btc_mir_5 = 1 if btc_return>0.05
gen btc_mir_10 = 1 if btc_return>0.1
gen btc_mir_20 = 1 if btc_return>0.2
gen btc_mir_30 = 1 if btc_return>0.3
gen dash_mir_5 = 1 if dash_return>0.05
gen dash_mir_10 = 1 if dash_return>0.1
gen dash_mir_20 = 1 if dash_return>0.2
gen dash_mir_30 = 1 if dash_return>0.3
gen eth_mir_5 = 1 if eth_return>0.05
gen eth_mir_10 = 1 if eth_return>0.1
gen eth_mir_20 = 1 if eth_return>0.2
gen eth_mir_30 = 1 if eth_return>0.3
gen ltc_mir_5 = 1 if ltc_return>0.05
gen ltc_mir_10 = 1 if ltc_return>0.1
gen ltc_mir_20 = 1 if ltc_return>0.2
gen ltc_mir_30 = 1 if ltc_return>0.3
egen total_btc_dis_5 = total(btc_dis_5)
egen total_btc_dis_10 = total(btc_dis_10)
egen total_btc_dis_20 = total(btc_dis_20)
egen total_btc_dis_30 = total(btc_dis_30)
egen total_btc_mir_5 = total(btc_mir_5)
egen total_btc_mir_10 = total(btc_mir_10)
egen total_btc_mir_20 = total(btc_mir_20)
egen total_btc_mir_30 = total(btc_mir_30)
egen total_dash_dis_5 = total(dash_dis_5)
egen total_dash_dis_10 = total(dash_dis_10)
egen total_dash_dis_20 = total(dash_dis_20)
egen total_dash_dis_30 = total(dash_dis_30)
egen total_dash_mir_5 = total(dash_mir_5)
egen total_dash_mir_10 = total(dash_mir_10)
egen total_dash_mir_20 = total(dash_mir_20)
egen total_dash_mir_30 = total(dash_mir_30)
egen total_eth_dis_5 = total(eth_dis_5)
egen total_eth_dis_10 = total(eth_dis_10)
egen total_eth_dis_20 = total(eth_dis_20)
egen total_eth_dis_30 = total(eth_dis_30)
egen total_eth_mir_5 = total(eth_mir_5)
egen total_eth_mir_10 = total(eth_mir_10)
egen total_eth_mir_20 = total(eth_mir_20)
egen total_eth_mir_30 = total(eth_mir_30)
egen total_ltc_dis_5 = total(ltc_dis_5)
egen total_ltc_dis_10 = total(ltc_dis_10)
egen total_ltc_dis_20 = total(ltc_dis_20)
egen total_ltc_dis_30 = total(ltc_dis_30)
egen total_ltc_mir_5 = total(ltc_mir_5)
egen total_ltc_mir_10 = total(ltc_mir_10)
egen total_ltc_mir_20 = total(ltc_mir_20)
egen total_ltc_mir_30 = total(ltc_mir_30)
save "1.dta", replace


######################### Factor models results ################################
use "1.dta", clear

* Figure 3 Price movements of Bitcoin
line btc_price tradingdate, graphregion(fcolor(white))

* Figure 4 Distribution of daily Bitcoin returns
hist btc_return, frequency bin(100) normal graphregion(fcolor(white)) vertical color("51 161 201"%20)

* Table 2 Summary statistics of the daily Bitcoin returns
** Panel A
summarize btc_return 
outreg2 using "Decriptive statistics.xls", replace sum(log) keep(btc_return) title(Decriptive statistics)
summarize btc_sharperatio 
summarize btc_return, detail
jb6 btc_return
sktest btc_return
di btc_r_positive/3349
** Panel B
di total_btc_dis_5
di total_btc_dis_10
di total_btc_dis_20
di total_btc_dis_30
di total_btc_mir_5
di total_btc_mir_10
di total_btc_mir_20
di total_btc_mir_30
di total_btc_dis_5/3349
di total_btc_dis_10/3349
di total_btc_dis_20/3349
di total_btc_dis_30/3349
di total_btc_mir_5/3349
di total_btc_mir_10/3349
di total_btc_mir_20/3349
di total_btc_mir_30/3349

* Table A2 Bitcoin's exposure to common factors
reg btc_return mkt_rf, r
outreg2 using "Table A2.xls", replace tstat bdec(3) tdec(2) ctitle(CAPM) 
reg btc_return mkt_rf smb hml, r
outreg2 using "Table A2.xls", append tstat bdec(3) tdec(2) ctitle(FF3) 
reg btc_return mkt_rf smb hml umd, r 
outreg2 using "Table A2.xls", append tstat bdec(3) tdec(2) ctitle(FFC) 
reg btc_return mkt_rf smb hml rmw cma, r
outreg2 using "Table A2.xls", append tstat bdec(3) tdec(2) ctitle(FF5) 
reg btc_return mkt_rf smb hml umd rmw cma, r
outreg2 using "Table A2.xls", append tstat bdec(3) tdec(2) ctitle(FF6) 

* Table A3 Bitcoin's exposure to currencies
reg btc_return US, r
outreg2 using "Table A3.xls", replace tstat bdec(3) tdec(2) ctitle(US) 
reg btc_return CAD, r
outreg2 using "Table A3.xls", append tstat bdec(3) tdec(2) ctitle(CAD) 
reg btc_return EUR, r
outreg2 using "Table A3.xls", append tstat bdec(3) tdec(2) ctitle(EUR) 
reg btc_return GBP, r
outreg2 using "Table A3.xls", append tstat bdec(3) tdec(2) ctitle(GBP) 
reg btc_return JPY, r
outreg2 using "Table A3.xls", append tstat bdec(3) tdec(2) ctitle(JPY) 
reg btc_return DKK, r
outreg2 using "Table A3.xls", append tstat bdec(3) tdec(2) ctitle(JPY) 

* Table A4 Bitcoin's sentiment
use btc_baidu_index, clear
rename index baidu
gen tradingdate = date(date,"YMD")
format tradingdate %td
sort tradingdate
keep tradingdate baidu
destring baidu, replace force
save baidu, replace
use btc_google_trends, clear
rename bitcoinworldwide google
gen tradingdate = date(day,"YMD")
format tradingdate %td
sort tradingdate
keep tradingdate google
destring google, replace force
save google, replace
merge m:m tradingdate using baidu
keep if _merge==3
drop _merge
merge m:m tradingdate using 1
sort tradingdate
keep if _merge==3
drop _merge
keep tradingdate btc_return baidu google
replace baidu = baidu/100000000
replace google = google/10000
reg btc_return l.baidu l.google, r
outreg2 using "Table A4.xls", replace tstat bdec(3) tdec(2) ctitle(R_t+1) 
reg btc_return l2.baidu l2.google, r
outreg2 using "Table A4.xls", append tstat bdec(3) tdec(2) ctitle(R_t+2) 
reg btc_return l3.baidu l3.google, r
outreg2 using "Table A4.xls", append tstat bdec(3) tdec(2) ctitle(R_t+3) 
reg btc_return l4.baidu l4.google, r
outreg2 using "Table A4.xls", append tstat bdec(3) tdec(2) ctitle(R_t+4) 
reg btc_return l5.baidu l5.google, r
outreg2 using "Table A4.xls", append tstat bdec(3) tdec(2) ctitle(R_t+5) 
reg btc_return l6.baidu l6.google, r
outreg2 using "Table A4.xls", append tstat bdec(3) tdec(2) ctitle(R_t+6) 
reg btc_return l7.baidu l7.google, r
outreg2 using "Table A4.xls", append tstat bdec(3) tdec(2) ctitle(R_t+7) 

* Table A5 Bitcoin's momentum
use "1.dta", clear
sort tradingdate
reg btc_return l.btc_return, r
outreg2 using "Table A5.xls", replace tstat bdec(3) tdec(2) ctitle(R_t+1) 
reg btc_return l2.btc_return, r
outreg2 using "Table A5.xls", append tstat bdec(3) tdec(2) ctitle(R_t+2) 
reg btc_return l3.btc_return, r
outreg2 using "Table A5.xls", append tstat bdec(3) tdec(2) ctitle(R_t+3) 
reg btc_return l4.btc_return, r
outreg2 using "Table A5.xls", append tstat bdec(3) tdec(2) ctitle(R_t+4) 
reg btc_return l5.btc_return, r
outreg2 using "Table A5.xls", append tstat bdec(3) tdec(2) ctitle(R_t+5) 
reg btc_return l6.btc_return, r
outreg2 using "Table A5.xls", append tstat bdec(3) tdec(2) ctitle(R_t+6) 
reg btc_return l7.btc_return, r
outreg2 using "Table A5.xls", append tstat bdec(3) tdec(2) ctitle(R_t+7) 

* Table A9-A12
use blockchain, clear
corr
use cryptocurrency, clear
corr
use sentiment, clear
corr
use macroeconomics, clear
corr