# Five training seeds: cell and paired statistics

Evaluation seeds1/2/3 are averaged within each training seed. Exact3125 paired resamples, individual percentile95 intervals. Missing cells after required early stop are not imputed.

| USD | phase | command | metric | s1 | s2 | s3 | s4 | s5 | mean |95%|
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
|symmetric|0|stand|20s signed drift deg|-4.24367|-2.47078|-28.2624|-12.2147|-1.36673|-9.71165|[-19.6737, -2.38374]|
|symmetric|0|stand|20s absolute drift deg|4.24367|2.47078|28.2624|12.2147|1.70441|9.77919|[2.51881, 19.7413]|
|symmetric|0|stand|startup0-2s deg|0.957862|0.141327|-2.52953|-1.09518|-1.30376|-0.765856|[-1.75235, 0.304634]|
|symmetric|0|stand|late5-20s deg/s|-0.288405|-0.145344|-1.43578|-0.617465|-0.0165876|-0.500717|[-0.988281, -0.122454]|
|symmetric|0|stand|left half-cycle deg|-1.01519|-1.50261|-0.205813|-0.681008|-0.721981|-0.825321|[-1.21016, -0.465173]|
|symmetric|0|stand|right half-cycle deg|0.930316|1.4532|-0.359435|0.436714|0.694646|0.631088|[0.0621978, 1.13947]|
|symmetric|0|stand|cycle sum deg|-0.0848734|-0.0494156|-0.565248|-0.244294|-0.0273346|-0.194233|[-0.393474, -0.0476747]|
|symmetric|0|stand|fall fraction|0|0|0|0|0|0|[0, 0]|
|symmetric|0|stand|initial exposure seconds|20|20|20|20|20|20|[20, 20]|
|symmetric|0|forward_03|20s signed drift deg|13.1054|41.8415|30.8543|11.1385|-18.8726|15.6134|[-2.92501, 33.0518]|
|symmetric|0|forward_03|20s absolute drift deg|13.1106|41.8415|30.8543|11.1392|20.1731|23.4238|[13.7345, 33.8979]|
|symmetric|0|forward_03|startup0-2s deg|-5.05487|0.506013|-4.79006|-4.81038|-6.44968|-4.11979|[-5.78989, -1.66538]|
|symmetric|0|forward_03|late5-20s deg/s|1.09713|2.32608|1.93921|0.920269|-0.431449|1.17025|[0.313026, 1.96755]|
|symmetric|0|forward_03|left half-cycle deg|0.781643|1.94987|0.895739|1.74884|1.13731|1.30268|[0.898414, 1.70694]|
|symmetric|0|forward_03|right half-cycle deg|-0.519536|-1.11304|-0.278653|-1.52607|-1.51476|-0.99041|[-1.43894, -0.541883]|
|symmetric|0|forward_03|cycle sum deg|0.262107|0.83683|0.617087|0.222769|-0.377452|0.312268|[-0.0585002, 0.661036]|
|symmetric|0|forward_03|fall fraction|0|0|0|0|0|0|[0, 0]|
|symmetric|0|forward_03|initial exposure seconds|20|20|20|20|20|20|[20, 20]|
|symmetric|0|forward_05|20s signed drift deg|-77.4198|11.9136|46.871|-23.4205|145.476|20.6841|[-41.6864, 91.9758]|
|symmetric|0|forward_05|20s absolute drift deg|77.4198|11.9174|46.871|23.4205|145.476|61.021|[23.5094, 106.034]|
|symmetric|0|forward_05|startup0-2s deg|-0.0828661|-0.41057|-1.38644|-1.3528|5.18686|0.390837|[-1.17781, 2.82498]|
|symmetric|0|forward_05|late5-20s deg/s|-4.36033|0.785623|3.18958|-1.20307|7.85728|1.25381|[-2.2189, 5.05666]|
|symmetric|0|forward_05|left half-cycle deg|0.713637|2.33124|3.85586|2.58668|2.72765|2.44301|[1.46285, 3.2971]|
|symmetric|0|forward_05|right half-cycle deg|-2.26203|-2.09297|-2.91844|-3.05509|0.181871|-2.02933|[-2.84182, -0.89316]|
|symmetric|0|forward_05|cycle sum deg|-1.5484|0.238271|0.937421|-0.468411|2.90952|0.413682|[-0.833729, 1.83952]|
|symmetric|0|forward_05|fall fraction|0|0|0|0|0|0|[0, 0]|
|symmetric|0|forward_05|initial exposure seconds|20|20|20|20|20|20|[20, 20]|
|symmetric|0|forward_08|20s signed drift deg|63.7929|91.739|26.0579|28.5839|52.227|52.4801|[32.3021, 73.5188]|
|symmetric|0|forward_08|20s absolute drift deg|63.7929|91.739|26.0634|28.5839|52.227|52.4813|[32.3043, 73.5188]|
|symmetric|0|forward_08|startup0-2s deg|-0.57044|5.97892|-1.79761|2.62506|6.43088|2.53336|[-0.422208, 5.48893]|
|symmetric|0|forward_08|late5-20s deg/s|3.69023|4.74096|1.51591|1.45289|2.51624|2.78325|[1.69077, 3.8858]|
|symmetric|0|forward_08|left half-cycle deg|3.7208|5.9457|5.03041|2.33118|-0.726086|3.2604|[1.05267, 5.1346]|
|symmetric|0|forward_08|right half-cycle deg|-2.44494|-4.11092|-4.50925|-1.7595|1.77063|-2.2108|[-3.93706, -0.11171]|
|symmetric|0|forward_08|cycle sum deg|1.27586|1.83478|0.521157|0.571679|1.04454|1.0496|[0.646042, 1.47038]|
|symmetric|0|forward_08|fall fraction|0|0|0|0|0|0|[0, 0]|
|symmetric|0|forward_08|initial exposure seconds|20|20|20|20|20|20|[20, 20]|
|symmetric|10|stand|20s signed drift deg|-7.11525|-2.96848|-28.528|-12.221|2.10971|-9.74459|[-19.9629, -1.76656]|
|symmetric|10|stand|20s absolute drift deg|7.11525|2.96848|28.528|12.221|2.19946|10.6064|[3.49023, 20.0009]|
|symmetric|10|stand|startup0-2s deg|-1.90869|-0.365637|-2.73765|-1.09193|2.04063|-0.812655|[-2.07936, 0.769514]|
|symmetric|10|stand|late5-20s deg/s|-0.289431|-0.145818|-1.43528|-0.617709|-0.0153389|-0.500714|[-0.987775, -0.122349]|
|symmetric|10|stand|left half-cycle deg|-1.0383|-1.4933|-0.218809|-0.679483|-0.684015|-0.822782|[-1.2203, -0.474843]|
|symmetric|10|stand|right half-cycle deg|0.895997|1.43393|-0.35175|0.435063|0.726209|0.62789|[0.079434, 1.11117]|
|symmetric|10|stand|cycle sum deg|-0.142305|-0.0593697|-0.570559|-0.24442|0.0421942|-0.194892|[-0.399257, -0.0353312]|
|symmetric|10|stand|fall fraction|0|0|0|0|0|0|[0, 0]|
|symmetric|10|stand|initial exposure seconds|20|20|20|20|20|20|[20, 20]|
|symmetric|10|forward_03|20s signed drift deg|18.9384|45.8347|41.0467|20.676|-14.3751|22.4241|[3.37195, 39.4978]|
|symmetric|10|forward_03|20s absolute drift deg|18.9384|45.8347|41.0467|20.676|17.6096|28.8211|[18.7544, 39.4978]|
|symmetric|10|forward_03|startup0-2s deg|-1.18605|3.75494|5.88729|3.83898|-1.74032|2.11097|[-0.513603, 4.6415]|
|symmetric|10|forward_03|late5-20s deg/s|1.10223|2.32156|1.92547|0.922411|-0.405418|1.17325|[0.326326, 1.96251]|
|symmetric|10|forward_03|left half-cycle deg|0.842376|2.01078|1.01551|1.84254|1.17641|1.37752|[0.978437, 1.77808]|
|symmetric|10|forward_03|right half-cycle deg|-0.463607|-1.09409|-0.19458|-1.42902|-1.46391|-0.929042|[-1.37599, -0.482093]|
|symmetric|10|forward_03|cycle sum deg|0.378768|0.916694|0.820934|0.41352|-0.287502|0.448483|[0.067439, 0.789957]|
|symmetric|10|forward_03|fall fraction|0|0|0|0|0|0|[0, 0]|
|symmetric|10|forward_03|initial exposure seconds|20|20|20|20|20|20|[20, 20]|
|symmetric|10|forward_05|20s signed drift deg|-85.0124|14.6886|46.8071|-27.1412|131.411|16.1506|[-45.132, 82.7798]|
|symmetric|10|forward_05|20s absolute drift deg|85.0124|14.6886|46.8071|27.1412|131.411|61.0121|[26.0933, 101.277]|
|symmetric|10|forward_05|startup0-2s deg|-7.16808|1.0025|-3.77378|-5.24296|-5.97756|-4.23198|[-6.30685, -1.49569]|
|symmetric|10|forward_05|late5-20s deg/s|-4.35505|0.803542|3.1886|-1.20443|7.87911|1.26236|[-2.2162, 5.06481]|
|symmetric|10|forward_05|left half-cycle deg|0.610338|2.3785|3.85128|2.53753|2.64356|2.40424|[1.38122, 3.29397]|
|symmetric|10|forward_05|right half-cycle deg|-2.31059|-2.08473|-2.91514|-3.08036|-0.0153381|-2.08123|[-2.86032, -1.00918]|
|symmetric|10|forward_05|cycle sum deg|-1.70025|0.293772|0.936141|-0.542824|2.62822|0.323012|[-0.902641, 1.6556]|
|symmetric|10|forward_05|fall fraction|0|0|0|0|0|0|[0, 0]|
|symmetric|10|forward_05|initial exposure seconds|20|20|20|20|20|20|[20, 20]|
|symmetric|10|forward_08|20s signed drift deg|61.7591|90.5887|35.0695|24.267|46.2559|51.588|[32.9858, 73.2909]|
|symmetric|10|forward_08|20s absolute drift deg|61.7591|90.5887|35.0695|24.274|46.2559|51.5894|[32.9886, 73.2909]|
|symmetric|10|forward_08|startup0-2s deg|-4.55032|4.52486|8.04044|-1.73547|1.8029|1.61648|[-2.17231, 5.3867]|
|symmetric|10|forward_08|late5-20s deg/s|3.69487|4.75927|1.51427|1.46624|2.51428|2.78979|[1.69506, 3.89739]|
|symmetric|10|forward_08|left half-cycle deg|3.70074|6.00088|5.01854|2.30991|-0.781668|3.24968|[1.01129, 5.14792]|
|symmetric|10|forward_08|right half-cycle deg|-2.46555|-4.18911|-4.31715|-1.82457|1.70678|-2.21792|[-3.89562, -0.178664]|
|symmetric|10|forward_08|cycle sum deg|1.23518|1.81177|0.701391|0.485341|0.925117|1.03176|[0.659716, 1.46582]|
|symmetric|10|forward_08|fall fraction|0|0|0|0|0|0|[0, 0]|
|symmetric|10|forward_08|initial exposure seconds|20|20|20|20|20|20|[20, 20]|

|USD|command|paired metric|s1|s2|s3|s4|s5|mean|95%|
|---|---|---|---:|---:|---:|---:|---:|---:|---|
|symmetric|stand|S20_degrees|-11.3589|-5.43926|-56.7903|-24.4357|0.742981|-19.4562|[-38.8127, -4.1503]|
|symmetric|stand|D20_degrees|1.43579|0.248852|0.132787|0.00312929|-1.73822|0.0164681|[-0.989816, 0.888658]|
|symmetric|stand|half_swap_left_residual_degrees|-0.107986|-0.0401007|-0.578244|-0.242769|0.0106311|-0.191694|[-0.393374, -0.0333851]|
|symmetric|stand|half_swap_right_residual_degrees|-0.119192|-0.0686846|-0.557563|-0.245945|0.00422855|-0.197431|[-0.382881, -0.0496209]|
|symmetric|stand|common_prefix_S_degrees|-11.3589|-5.43926|-56.7903|-24.4357|0.742981|-19.4562|[-38.8127, -4.1503]|
|symmetric|stand|common_prefix_D_degrees|1.43579|0.248852|0.132787|0.00312929|-1.73822|0.0164681|[-0.989816, 0.888658]|
|symmetric|stand|common_prefix_seconds|20|20|20|20|20|20|[20, 20]|
|symmetric|stand|paired_complete20s_count|1024|1024|1024|1024|1024|1024|[1024, 1024]|
|symmetric|forward_03|S20_degrees|32.0438|87.6762|71.901|31.8145|-33.2477|38.0375|[0.794463, 73.3488]|
|symmetric|forward_03|D20_degrees|-2.91652|-1.99659|-5.09618|-4.76877|-2.24875|-3.40536|[-4.52928, -2.28144]|
|symmetric|forward_03|half_swap_left_residual_degrees|0.32284|0.897747|0.736861|0.316475|-0.338351|0.387114|[0.00765628, 0.749315]|
|symmetric|forward_03|half_swap_right_residual_degrees|0.318036|0.855777|0.70116|0.319815|-0.326603|0.373637|[0.00787715, 0.717305]|
|symmetric|forward_03|common_prefix_S_degrees|32.0438|87.6762|71.901|31.8145|-33.2477|38.0375|[0.794463, 73.3488]|
|symmetric|forward_03|common_prefix_D_degrees|-2.91652|-1.99659|-5.09618|-4.76877|-2.24875|-3.40536|[-4.52928, -2.28144]|
|symmetric|forward_03|common_prefix_seconds|20|20|20|20|20|20|[20, 20]|
|symmetric|forward_03|paired_complete20s_count|1024|1024|1024|1024|1024|1024|[1024, 1024]|
|symmetric|forward_05|S20_degrees|-162.432|26.6022|93.6781|-50.5617|276.887|36.8347|[-86.8185, 174.756]|
|symmetric|forward_05|D20_degrees|3.79633|-1.38751|0.0319866|1.86033|7.03253|2.26673|[-0.170143, 4.98518]|
|symmetric|forward_05|half_swap_left_residual_degrees|-1.65169|0.285524|0.932838|-0.517556|2.82543|0.374909|[-0.876807, 1.77831]|
|symmetric|forward_05|half_swap_right_residual_degrees|-1.59695|0.246519|0.940723|-0.493678|2.71231|0.361785|[-0.859563, 1.7168]|
|symmetric|forward_05|common_prefix_S_degrees|-162.432|26.6022|93.6781|-50.5617|276.887|36.8347|[-86.8185, 174.756]|
|symmetric|forward_05|common_prefix_D_degrees|3.79633|-1.38751|0.0319866|1.86033|7.03253|2.26673|[-0.170143, 4.98518]|
|symmetric|forward_05|common_prefix_seconds|20|20|20|20|20|20|[20, 20]|
|symmetric|forward_05|paired_complete20s_count|1024|1024|1024|1024|1024|1024|[1024, 1024]|
|symmetric|forward_08|S20_degrees|125.552|182.328|61.1274|52.851|98.4828|104.068|[65.2879, 146.733]|
|symmetric|forward_08|D20_degrees|1.01693|0.575165|-4.50583|2.15846|2.98555|0.446054|[-2.15678, 2.32388]|
|symmetric|forward_08|half_swap_left_residual_degrees|1.2558|1.88996|0.509295|0.550404|0.988957|1.03888|[0.621671, 1.49522]|
|symmetric|forward_08|half_swap_right_residual_degrees|1.25524|1.75659|0.713252|0.506615|0.980699|1.04248|[0.684087, 1.44766]|
|symmetric|forward_08|common_prefix_S_degrees|125.552|182.328|61.1274|52.851|98.4828|104.068|[65.2879, 146.733]|
|symmetric|forward_08|common_prefix_D_degrees|1.01693|0.575165|-4.50583|2.15846|2.98555|0.446054|[-2.15678, 2.32388]|
|symmetric|forward_08|common_prefix_seconds|20|20|20|20|20|20|[20, 20]|
|symmetric|forward_08|paired_complete20s_count|1024|1024|1024|1024|1024|1024|[1024, 1024]|
