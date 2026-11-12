# Frozen30degree heading gate applied to issue15 data

Zero re-evaluation of the lineage. Endpoint means remain conditional on no-reset20s survivors; prefixes/falls are never treated as full20s endpoints. Upper95 is a separately reported conservative flag, not a changed primary mean gate. Single-training-seed cells have no training CI.

|Node|Command|Train n|Survived / originalN|Mean abs degrees|Training95 upper|Mean<=30|Endpoint scope|
|---|---|---:|---:|---:|---:|---|---|
|ft5|stand|1|3072/3072|3.6326499473759903|None|True|complete_no_reset20s_endpoint|
|ft5|forward_03|1|3072/3072|92.94448352198208|None|False|complete_no_reset20s_endpoint|
|ft5|forward_05|1|322/3072|50.79702077140814|None|False|survivor_selected_endpoint|
|ft5|forward_08|1|0/3072|None|None|False|missing_or_censored_in_some_seed|
|robust05|stand|1|3072/3072|16.922188399748023|None|True|complete_no_reset20s_endpoint|
|robust05|forward_03|1|3072/3072|2.4036730414872207|None|True|complete_no_reset20s_endpoint|
|robust05|forward_05|1|3072/3072|19.708534581283974|None|True|complete_no_reset20s_endpoint|
|robust05|forward_08|1|0/3072|None|None|False|missing_or_censored_in_some_seed|
|robust08|stand|1|3072/3072|0.45728270980759006|None|True|complete_no_reset20s_endpoint|
|robust08|forward_03|1|3072/3072|16.29964459586118|None|True|complete_no_reset20s_endpoint|
|robust08|forward_05|1|3072/3072|14.593718162748681|None|True|complete_no_reset20s_endpoint|
|robust08|forward_08|1|3072/3072|32.439412692126375|None|False|complete_no_reset20s_endpoint|
|strengthen|stand|2|6144/6144|48.79628916177141|88.42109439557787|False|complete_no_reset20s_endpoint|
|strengthen|forward_03|2|6144/6144|42.56308183688405|79.75689659655586|False|complete_no_reset20s_endpoint|
|strengthen|forward_05|2|6144/6144|69.3593194580883|120.40600984977156|False|complete_no_reset20s_endpoint|
|strengthen|forward_08|2|6144/6144|66.77344968180718|125.39476448144627|False|complete_no_reset20s_endpoint|
|bounded|stand|2|6144/6144|31.854108431501388|54.818982035193265|False|complete_no_reset20s_endpoint|
|bounded|forward_03|2|6144/6144|76.32377869472289|114.19627531265031|False|complete_no_reset20s_endpoint|
|bounded|forward_05|2|6144/6144|233.54908388655235|250.14666218153187|False|complete_no_reset20s_endpoint|
|bounded|forward_08|2|6144/6144|251.19440615423633|316.5003232709075|False|complete_no_reset20s_endpoint|
|requested1500|stand|5|15360/15360|38.24737215204511|53.185701002987095|False|complete_no_reset20s_endpoint|
|requested1500|forward_03|5|15360/15360|75.66371163765936|109.68329877353104|False|complete_no_reset20s_endpoint|
|requested1500|forward_05|5|15360/15360|200.13551486359444|255.4256576864957|False|complete_no_reset20s_endpoint|
|requested1500|forward_08|5|15360/15360|245.70077550236084|338.2500031673089|False|complete_no_reset20s_endpoint|
|requested3000|stand|5|15360/15360|37.58669887069061|69.24184134964074|False|complete_no_reset20s_endpoint|
|requested3000|forward_03|5|15360/15360|121.47154301081324|164.2924769237679|False|complete_no_reset20s_endpoint|
|requested3000|forward_05|5|15360/15360|346.27464684727573|400.78112620973803|False|complete_no_reset20s_endpoint|
|requested3000|forward_08|5|15360/15360|506.03591122013603|610.0348559384262|False|complete_no_reset20s_endpoint|
|requested4500|stand|5|15360/15360|65.99880660589095|95.59169328034382|False|complete_no_reset20s_endpoint|
|requested4500|forward_03|5|15360/15360|243.3770153829646|306.7838538634908|False|complete_no_reset20s_endpoint|
|requested4500|forward_05|5|15360/15360|415.3139011711557|464.5408595944831|False|complete_no_reset20s_endpoint|
|requested4500|forward_08|5|15360/15360|527.2038549399917|627.5649819346916|False|complete_no_reset20s_endpoint|
|control6000|stand|5|15360/15360|29.471023692969187|41.73909191206653|True|complete_no_reset20s_endpoint|
|control6000|forward_03|5|15360/15360|107.1073677503739|154.39161283126617|False|complete_no_reset20s_endpoint|
|control6000|forward_05|5|15360/15360|254.59611658626983|299.56446465451614|False|complete_no_reset20s_endpoint|
|control6000|forward_08|5|15360/15360|311.896575778845|398.3408843497609|False|complete_no_reset20s_endpoint|
|treatment|stand|5|15360/15360|73.61380768725391|124.12701520061103|False|complete_no_reset20s_endpoint|
|treatment|forward_03|5|15360/15360|238.54694747306021|287.92822206226685|False|complete_no_reset20s_endpoint|
|treatment|forward_05|5|15360/15360|336.65761523817525|487.8372025001505|False|complete_no_reset20s_endpoint|
|treatment|forward_08|5|15360/15360|406.9589195106209|544.9479893559372|False|complete_no_reset20s_endpoint|
|clock|stand|5|15360/15360|57.51525770037513|90.8644739687873|False|complete_no_reset20s_endpoint|
|clock|forward_03|5|15360/15360|167.0481234732091|193.58012257674537|False|complete_no_reset20s_endpoint|
|clock|forward_05|5|15360/15360|272.5210665998895|374.00178044324906|False|complete_no_reset20s_endpoint|
|clock|forward_08|5|15360/15360|447.4038399816318|552.8293946458793|False|complete_no_reset20s_endpoint|

All-four mean-gate node flags: {"ft5": false, "robust05": false, "robust08": false, "strengthen": false, "bounded": false, "requested1500": false, "requested3000": false, "requested4500": false, "control6000": false, "treatment": false, "clock": false}

All signed training-seed values, prefix drift/duration and reset-excluded segment metrics remain in lineage_heading_gate.json. A negative group mean is not evidence that every seed or trial turns the same direction.
