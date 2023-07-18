#undef __IF_FP64

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define __IF_FP64(X) X
#else
#define __IF_FP64(X)
#endif

#undef __IF_INT64
#if defined(__opencl_c_int64) || defined(cl_khr_int64)
#define __IF_INT64(X) X
#else
#define __IF_INT64(X)
#endif

#ifndef _CL_NOINLINE
#define _CL_NOINLINE __attribute__((__noinline__))
#endif

#ifndef _CL_OVERLOADABLE
#define _CL_OVERLOADABLE __attribute__((__overloadable__))
#endif

constant float values[] = {
  0.0f,     0.1f,      0.9f,        1.0f,
  1.1f,     10.0f,     1000000.0f,  1000000000000.0f,
  MAXFLOAT, HUGE_VALF, INFINITY,    (0.0f / 0.0f),
  FLT_MAX,  FLT_MIN,   FLT_EPSILON,
};
constant int nvalues = sizeof (values) / sizeof (*values);
#ifdef cl_khr_fp64
constant double dvalues[] = {
  0.0,
  0.1,
  0.9,
  1.0,
  1.1,
  10.0,
  1000000.0,
  1000000000000.0,
  1000000000000000000000000.0,
  HUGE_VAL,
  INFINITY,
  (0.0f / 0.0f),
  DBL_MAX,
  DBL_MIN,
  DBL_EPSILON,
};
constant int ndvalues = sizeof (dvalues) / sizeof (*dvalues);
constant int ninputs = 2;
#else
constant int ninputs = 1;
#endif

   constant uint randoms[] = {
     0x00000000U,
     0x00000001U,
     0x7fffffffU,
     0x80000000U,
     0xfffffffeU,
     0xffffffffU,
     0x01010101U,
     0x80808080U,
     0x55555555U,
     0xaaaaaaaaU,
     116127149U,
     331473970U,
     3314285513U,
     1531519032U,
     3871781304U,
     723260354U,
     3734992454U,
     3048883544U,
     424075405U,
     3760586679U,
     364071113U,
     2212396745U,
     3026460845U,
     2062923368U,
     3945483116U,
     774301702U,
     2010645213U,
     353497300U,
     2240089293U,
     645959945U,
     2929402380U,
     3641106046U,
     3731530029U,
     3788502454U,
     3990366079U,
     3532452335U,
     3231247251U,
     123690193U,
     418692672U,
     4146745661U,
     4170087687U,
     3915754726U,
     2052700648U,
     1748863847U,
     276568793U,
     364266289U,
     24718041U,
     3775186845U,
     935438421U,
     3070232227U,
     558364671U,
     2318351214U,
     17943242U,
     1796864907U,
     727165514U,
     223478118U,
     2448924107U,
     496915291U,
     3372891854U,
     361433487U,
     3273766229U,
     251831411U,
     432661417U,
     772908669U,
     289792578U,
     4150526710U,
     4157662725U,
     2594757327U,
     3052388893U,
     3842089578U,
     3467269013U,
     510187125U,
     2596093643U,
     398042620U,
     4272455984U,
     3711648086U,
     2120827851U,
     77269246U,
     2168059317U,
     2750549452U,
     1712682330U,
     2486520097U,
     625173621U,
     1632501477U,
     2935468416U,
     980045574U,
     3080136685U,
     4291385683U,
     1900746145U,
     3343063222U,
     3737266887U,
     3349055009U,
     3557165116U,
     847440541U,
     1195278641U,
     313889830U,
     622790046U,
     326637691U,
     663570370U,
     662327410U,
     923839117U,
     3091793818U,
     3563062752U,
     1864236072U,
     4251970867U,
     2259486024U,
     2512789432U,
     4278284968U,
     244581614U,
     247706675U,
     3268622648U,
     3758387026U,
     206893256U,
     2892198447U,
     3585538105U,
     2484801188U,
     1063964031U,
     3712657639U,
     23179627U,
     1732005357U,
     2522016557U,
     1058341654U,
     1580368080U,
     1890361257U,
     1167428989U,
     2600065453U,
     1547136389U,
     945856727U,
     2005682606U,
     3399854093U,
     2619154565U,
     2207015138U,
     2836381097U,
     612928932U,
     1537934908U,
     897756908U,
     1142275256U,
     1106163744U,
     3209429231U,
     3317761168U,
     2815958850U,
     1282374282U,
     3861163766U,
     2547903564U,
     3139840265U,
     587243656U,
     3261127556U,
     3955999184U,
     2061849860U,
     3778058575U,
     259659645U,
     935157504U,
     3294850933U,
     2164603733U,
     3772888022U,
     732201413U,
     3677934092U,
     321204420U,
     509807651U,
     3626474557U,
     284622251U,
     3655952885U,
     1512028769U,
     1102588652U,
     2700179235U,
     4167405174U,
     2672050627U,
     3410780487U,
     4153733940U,
     2459759898U,
     568792515U,
     1081882827U,
     3211871042U,
     799411732U,
     2101993855U,
     3415550991U,
     3872737342U,
     4168312654U,
     1889019671U,
     4247531636U,
     2442118552U,
     3024016549U,
     1041817509U,
     141773691U,
     28033810U,
     4034097901U,
     1532981240U,
     2593712697U,
     2751535537U,
     269072724U,
     3363560906U,
     3555817938U,
     611297346U,
     366972507U,
     788151801U,
     3990920857U,
     1611303958U,
     3353102293U,
     1334246396U,
     1114446428U,
     3491128109U,
     2922751152U,
     3053407478U,
     2897830841U,
     176546593U,
     3184221063U,
     37923477U,
     1692128510U,
     165719856U,
     1795746307U,
     2422422413U,
     253227286U,
     2188522595U,
     582156087U,
     2342528685U,
     2080142547U,
     1928462563U,
     2713927482U,
     1944972771U,
     2534268146U,
     830798003U,
     1653357460U,
     291743070U,
     593771532U,
     2941865444U,
     855254640U,
     2401129822U,
     2420945774U,
     2447532144U,
     1137540092U,
     1296659939U,
     3252539825U,
     1165427708U,
     3251476781U,
     2597490804U,
     2518198923U,
     1196242486U,
     3646082981U,
     1347758965U,
     3824891532U,
     2959519286U,
     1523237529U,
     2910666174U,
     3226637035U,
     2116458903U,
     1076998092U,
     4222762545U,
     3061300520U,
     4189298288U,
     3943996060U,
     3129210496U,
     3826669630U,
     4235952488U,
     2624429853U,
     2522766390U,
     4137227001U,
     3846448057U,
     1893377487U,
     3658784739U,
     2368074586U,
     170547540U,
     520741120U,
     2662229630U,
     4265731754U,
     1379762094U,
     3395502906U,
     2242123335U,
     1960965916U,
     561815223U,
     2687853297U,
     4051050259U,
     1845906614U,
     3725623071U,
     1857706909U,
     2487006596U,
     1925919247U,
     2796536825U,
     3499954730U,
     2173320675U,
     3416676849U,
     3637473517U,
     340951464U,
     4152841543U,
     3747544606U,
     2659955417U,
     1695145107U,
     3117280269U,
     826143012U,
     3867179892U,
     4269349771U,
     1002613766U,
     3842086144U,
     1431990957U,
     2466205499U,
     653575141U,
     293530756U,
     2318035308U,
     3728576309U,
     1697894989U,
     2955143882U,
     2109912287U,
     2764187839U,
     1805490664U,
     672567480U,
     1374741155U,
     1662665091U,
     3551530257U,
     350283994U,
     685023916U,
     1887748803U,
     1386316091U,
     185708823U,
     3106823178U,
     3014109065U,
     3823816879U,
     2213358313U,
     2696977340U,
     4075569311U,
     365089277U,
     3466850767U,
     312392153U,
     1065191758U,
     2405243644U,
     3174745999U,
     3617861250U,
     867192904U,
     1046475095U,
     1888985494U,
     1127140157U,
     61671281U,
     128055546U,
     2332619657U,
     993669439U,
     2145370329U,
     1462433204U,
     74990676U,
     2898191247U,
     3601586977U,
     794604597U,
     3597643629U,
     4282141339U,
     251591051U,
     84943504U,
     2016044077U,
     946823499U,
     648214756U,
     2530104367U,
     4254219656U,
     1974542801U,
     53097687U,
     157109688U,
     299310673U,
     2866882336U,
     3335682769U,
     2583612755U,
     4114730718U,
     740387484U,
     986157357U,
     1140355266U,
     2825639379U,
     1198731547U,
     1521261313U,
     1204836445U,
     4294274455U,
     2215732661U,
     1369520150U,
     1515223958U,
     2428295267U,
     1945985266U,
     2168529560U,
     3791933294U,
     4021389338U,
     713695045U,
     4254483898U,
     3795986293U,
     1347498014U,
     1746051095U,
     1364967734U,
     206265390U,
     3940088473U,
     1867270033U,
     3893545471U,
     3545819698U,
     2573105187U,
     3859595967U,
     2823745089U,
     1293424244U,
     3948799370U,
     1524394803U,
     3807487752U,
     4055830971U,
     3124609223U,
     119357574U,
     1490516894U,
     3799908122U,
     1700941394U,
     80878888U,
     2719184407U,
     3603450215U,
     27225525U,
     1413638246U,
     3350206268U,
     2643568519U,
     801305037U,
     1341902999U,
     1420459209U,
     968648411U,
     1826125841U,
     2619721007U,
     537879916U,
     860253620U,
     586683700U,
     827412286U,
     2724526294U,
     1019678576U,
     3998975225U,
     339789397U,
     863181640U,
     970475690U,
     2737385140U,
     322021174U,
     4084948327U,
     80691950U,
     1702782677U,
     1266230197U,
     1100861683U,
     3123418948U,
     258978579U,
     3217833394U,
     1780903315U,
     1345341356U,
     2927579299U,
     931392918U,
     9404798U,
     83278219U,
     2470714323U,
     640357359U,
     2169696414U,
     496463525U,
     4127940882U,
     2965369765U,
     4136333330U,
     1159134689U,
     1798163043U,
     4097403856U,
     4284804850U,
     3165524545U,
     2765224926U,
     931350022U,
     1171636623U,
     845799406U,
     709853915U,
     2348457302U,
     3343956878U,
     2438786363U,
     175730452U,
     598587430U,
     2744955366U,
     447049527U,
     1252796590U,
     3044128900U,
     812683575U,
     3721040746U,
     3404688504U,
     2674021068U,
     959056069U,
     322162714U,
     2008064015U,
     3758321185U,
     2877937989U,
     778007512U,
     3502772435U,
     3084124565U,
     111844966U,
     248248909U,
     22147113U,
     2506501875U,
     1430033847U,
     1690841637U,
     2999017281U,
     3658748205U,
     1632773934U,
     4177069459U,
     3187781304U,
     1182255965U,
     4121685939U,
     300554973U,
     2854502901U,
     642657206U,
     1504346771U,
     128405037U,
     2163092164U,
     1091806675U,
     1144089805U,
     54479906U,
     505543118U,
     2844153548U,
     1010229282U,
     2961721580U,
     4235612700U,
     3508832243U,
     1409461040U,
     2568735295U,
     1191284023U,
     2220949766U,
     2605559386U,
     706551146U,
     3452279268U,
     2372892169U,
     2360210709U,
     3228881405U,
     2987444766U,
     1187314024U,
     908783041U,
     144096950U,
     1915948100U,
     2171208878U,
     420772043U,
     793209353U,
     359527746U,
     625018196U,
     1195796799U,
     2079388581U,
     864869238U,
     765565143U,
     1069647859U,
     3857355469U,
     2436437044U,
     238157644U,
     1612883577U,
     1911189891U,
     2070273440U,
     384222456U,
     1186369477U,
     2844794758U,
     3435869876U,
     1486894286U,
     4062343990U,
     440437688U,
     306253241U,
     3650751868U,
     2695961920U,
     3920128930U,
     3921419250U,
     502951143U,
     311093469U,
     2708936678U,
     36677206U,
     3473343884U,
     577655290U,
     3795127787U,
     1448118037U,
     436359554U,
     2051970204U,
     2644913053U,
     2492587228U,
     3125803824U,
     150160619U,
     1725373463U,
     2221292372U,
     2580064663U,
     1330289179U,
     2700556441U,
     1327212925U,
     651999045U,
     2089310372U,
     3221246949U,
     4148251434U,
     4267892623U,
     897583443U,
     1051813251U,
     2131903377U,
     4121163297U,
     4128279241U,
     1634689556U,
     3369895626U,
     1121895497U,
     3158192590U,
     4290462018U,
     3447288838U,
     4035505534U,
     2945114940U,
     1556028368U,
     4235061319U,
     1535570089U,
     2144940257U,
     1961364931U,
     2509075082U,
     804411045U,
     2290609740U,
     1076471626U,
     3254493188U,
     4284011230U,
     923006875U,
     3722016670U,
     2981439178U,
     2038308778U,
     1755166344U,
     488581856U,
     2624361425U,
     1298790575U,
     3550671725U,
     1845109437U,
     2047411775U,
     2488464246U,
     1391825885U,
     2340290304U,
     3623879917U,
     217171099U,
     3698905333U,
     2718846041U,
     73731529U,
     2053405441U,
     2770197347U,
     2983996080U,
     2612966141U,
     2187183079U,
     2796212469U,
     3797629169U,
     1788932364U,
     17748377U,
     627297271U,
     3689459731U,
     3311799950U,
     4263162298U,
     4016852324U,
     3136750215U,
     1725824049U,
     2844064064U,
     2059159211U,
     3182127070U,
     470655679U,
     1166949584U,
     2425843062U,
     219908183U,
     161770982U,
     2394961157U,
     999226372U,
     2367624166U,
     76287885U,
     1110832227U,
     3358123709U,
     1504127646U,
     49596774U,
     1296560019U,
     2320978173U,
     1163934122U,
     1631947491U,
     2702852639U,
     3856755518U,
     2562943123U,
     991330989U,
     993726248U,
     2133737192U,
     20974150U,
     3808389889U,
     2447868340U,
     2434828629U,
     3344419509U,
     4076789444U,
     1446054487U,
     3815933708U,
     3644670988U,
     3175898122U,
     3057844745U,
     559106380U,
     1840065631U,
     3020573012U,
     3203040371U,
     997381925U,
     2563312032U,
     815510593U,
     121805231U,
     1047507862U,
     1841403695U,
     1563170561U,
     1644198099U,
     3470882735U,
     627296501U,
     3006157508U,
     383648566U,
     3136652449U,
     2252034149U,
     1749861990U,
     956381402U,
     3299624735U,
     2798395931U,
     270054444U,
     3757564211U,
     2933717597U,
     1080178310U,
     1367392714U,
     1135266342U,
     2642448461U,
     1067554284U,
     3694982777U,
     3594374699U,
     4170301369U,
     3593401570U,
     2298071009U,
     1561680798U,
     2788490866U,
     1757829499U,
     8819607U,
     2453686068U,
     3458682663U,
     1614888171U,
     2327536307U,
     13960177U,
     125752716U,
     2312371195U,
     1515197240U,
     189747227U,
     666988376U,
     1401118738U,
     986465965U,
     242793663U,
     1830586663U,
     1603054176U,
     391536104U,
     1403125754U,
     4021998614U,
     157985039U,
     966292223U,
     2476444819U,
     3261614719U,
     3888752449U,
     2300656903U,
     1138839559U,
     1227396086U,
     1029493665U,
     2138482384U,
     2182525175U,
     1437393012U,
     2758514342U,
     1394715363U,
     242430786U,
     4026759135U,
     379455166U,
     3454852592U,
     1128257576U,
     513994046U,
     2437643547U,
     1851772774U,
     1096918785U,
     2537378072U,
     2020382559U,
     1306056753U,
     519939769U,
     2477462755U,
     2962076712U,
     2856059355U,
     111272034U,
     2363778749U,
     3031510224U,
     297098997U,
     2716928589U,
     1988398361U,
     3715685207U,
     1158387390U,
     3239718824U,
     214276640U,
     1240159361U,
     302800084U,
     258391670U,
     3118615408U,
     1789752935U,
     935790045U,
     1678444383U,
     3645357112U,
     1752731774U,
     1211889371U,
     2432949496U,
     1983838022U,
     2563701701U,
     3235972690U,
     2732559614U,
     4173627589U,
     918129740U,
     3528101943U,
     945287787U,
     783593046U,
     1687101911U,
     4265659819U,
     1625936204U,
     419423123U,
     404748783U,
     174814826U,
     561306387U,
     441376876U,
     3649973873U,
     1191532754U,
     493829681U,
     462640703U,
     3037639795U,
     4234288143U,
     787992128U,
     354556603U,
     1391557094U,
     1227150157U,
     25592400U,
     3032298621U,
     1655829692U,
     1736544192U,
     2936173068U,
     1867683432U,
     3284761215U,
     2988749127U,
     62083315U,
     3675433852U,
     1134152479U,
     2537382040U,
     1147996351U,
     1287284159U,
     1889610942U,
     3549411223U,
     2634772335U,
     1621708033U,
     3268420142U,
     2635222095U,
     2856377255U,
     3703296204U,
     45831019U,
     1997278369U,
     1472530726U,
     4202051236U,
     1958581642U,
     1899513707U,
     1642075765U,
     217373156U,
     1177071505U,
     2179831909U,
     1894821896U,
     375785474U,
     140181353U,
     2743987480U,
     123627609U,
     3644816362U,
     4244769687U,
     4053481902U,
     4272740073U,
     1701735471U,
     1799303028U,
     2810175160U,
     1531107068U,
     3059813822U,
     4125025775U,
     1932301928U,
     358163550U,
     1246286294U,
     1901878857U,
     2449370117U,
     4061706076U,
     2875797072U,
     1661522553U,
     543545982U,
     300448222U,
     4019581644U,
     3197346443U,
     731278538U,
     457112622U,
     669625172U,
     2548620393U,
     2931934447U,
     2318225955U,
     427149964U,
     1097556601U,
     3585697077U,
     1901391738U,
     3019912350U,
     4193989774U,
     1411691495U,
     2549773310U,
     3130489018U,
     739444137U,
     1953561922U,
     228589899U,
     974825144U,
     1873934953U,
     918502475U,
     4020302125U,
     2103082289U,
     1474428456U,
     269315616U,
     3376419786U,
     2903506696U,
     169344159U,
     4151327830U,
     2861975985U,
     1583628545U,
     337656074U,
     2381206238U,
     1346357469U,
     3316549550U,
     1188140897U,
     928463634U,
     120466083U,
     1048016215U,
     2053770646U,
     3729204448U,
     3630812747U,
     3421817962U,
     1471357089U,
     2971633393U,
     2721366758U,
     3977792328U,
     2771228423U,
     258029855U,
     325097628U,
     2816869331U,
     228010778U,
     1815596248U,
     2677647806U,
     4069826588U,
     2009464559U,
     4003870353U,
     2558198381U,
     823508134U,
     256895388U,
     130455482U,
     4107398577U,
     2446165146U,
     3086759840U,
     3128842794U,
     236454548U,
     3740649072U,
     1049081391U,
     3780795812U,
     1964380357U,
     3900635454U,
     1941196066U,
     1143285596U,
     1276856333U,
     2919547816U,
     2947639569U,
     1889305089U,
     2386910172U,
     2685680362U,
     2042792556U,
     2780968041U,
     976912013U,
     3562274424U,
     2336140155U,
     3464857244U,
     1108365812U,
     1201566469U,
     707126700U,
     4047776595U,
     1289380202U,
     1231913128U,
     2819729319U,
     537908270U,
     3802355886U,
     2004615093U,
     2947614997U,
     4192189156U,
     2809733754U,
     3082820238U,
     2758499499U,
     1004612882U,
     1102702383U,
     1862546275U,
     3170345990U,
     883739952U,
     1641198615U,
     957782688U,
     1503652889U,
     2210400768U,
     2002162781U,
     1553086024U,
     2591721606U,
     3830165160U,
     4181044959U,
     2735782270U,
     3825677158U,
     143739895U,
     771193452U,
     35990560U,
     1014009970U,
     20768744U,
     1785268932U,
     1424740580U,
     1620237280U,
     848157259U,
     3808893671U,
     2746756110U,
     3903639825U,
     1822084165U,
     2891666588U,
     3853186896U,
     4248495212U,
     1178592425U,
     455721495U,
     1848821934U,
     1558397701U,
     133397899U,
     1845531767U,
     2798312897U,
     1471176399U,
     1743248506U,
     2229972777U,
     1290369879U,
     3579075953U,
     309034994U,
     929728690U,
     3841454719U,
     3031753515U,
     3606461413U,
     2412281758U,
     2993123515U,
   };
   constant int nrandoms = sizeof(randoms) / sizeof(*randoms);



typedef constant char *string;

#ifdef cl_khr_fp64

#define ZERO_INPUT                                                            \
  if (input == 0)                                                             \
    {                                                                         \
      val.s[n] = sign * values[(iter + n) % nvalues];                         \
      good.s[n] = values[(iter + n) % nvalues];                               \
      val2.s[n] = values[(iter + n + 1) % nvalues];                           \
    }                                                                         \
  else                                                                        \
    {                                                                         \
      val.s[n] = sign * dvalues[(iter + n) % ndvalues];                       \
      good.s[n] = dvalues[(iter + n) % ndvalues];                             \
      val2.s[n] = dvalues[(iter + n + 1) % ndvalues];                         \
    }

#else

#define ZERO_INPUT                                                            \
  if (input == 0)                                                             \
    {                                                                         \
      val.s[n] = sign * values[(iter + n) % nvalues];                         \
      good.s[n] = values[(iter + n) % nvalues];                               \
      val2.s[n] = values[(iter + n + 1) % nvalues];                           \
    }

#endif

#define IMPLEMENT_BODY_V(NAME, BODY, SIZE, VTYPE, STYPE, ITYPE, SITYPE,       \
                         JTYPE, SJTYPE)                                       \
  void NAME##_##VTYPE ()                                                      \
  {                                                                           \
    typedef VTYPE vtype;                                                      \
    typedef STYPE stype;                                                      \
    typedef ITYPE itype;                                                      \
    typedef SITYPE sitype;                                                    \
    typedef JTYPE jtype;                                                      \
    typedef SJTYPE sjtype;                                                    \
    string const typename = #VTYPE;                                           \
    const int vecsize = SIZE;                                                 \
    typedef union                                                             \
    {                                                                         \
      vtype v;                                                                \
      volatile stype s[16];                                                   \
      /* s is declared as volatile to force all accesses through memory,      \
       * avoiding excess precision from x87 fpu registers                     \
       * see https://github.com/pocl/pocl/issues/621,                         \
       * https://gcc.gnu.org/wiki/x87note                                     \
       */                                                                     \
    } Tvec;                                                                   \
    Tvec val, good, val2;                                                     \
    Tvec res;                                                                 \
    bool equal;                                                               \
    typedef union                                                             \
    {                                                                         \
      stype s;                                                                \
      sjtype sj;                                                              \
    } S;                                                                      \
    typedef union                                                             \
    {                                                                         \
      itype v;                                                                \
      sitype s[16];                                                           \
    } Ivec;                                                                   \
    typedef union                                                             \
    {                                                                         \
      jtype v;                                                                \
      sjtype s[16];                                                           \
    } Jvec;                                                                   \
    for (int input = 0; input < ninputs; ++input)                             \
      {                                                                       \
        for (int iter = 0; iter < nvalues; ++iter)                            \
          {                                                                   \
            for (int sign = -1; sign <= +1; sign += 2)                        \
              {                                                               \
                for (int n = 0; n < vecsize; ++n)                             \
                  {                                                           \
                    ZERO_INPUT;                                               \
                  }                                                           \
                                                                              \
                BODY;                                                         \
              }                                                               \
          }                                                                   \
      }                                                                       \
  }

#define DEFINE_BODY_V(NAME, EXPR)                                             \
  IMPLEMENT_BODY_V (NAME, EXPR, 1, float, float, int, int, int, int)          \
  IMPLEMENT_BODY_V (NAME, EXPR, 2, float2, float, int2, int, int2, int)       \
  IMPLEMENT_BODY_V (NAME, EXPR, 3, float3, float, int3, int, int3, int)       \
  IMPLEMENT_BODY_V (NAME, EXPR, 4, float4, float, int4, int, int4, int)       \
  IMPLEMENT_BODY_V (NAME, EXPR, 8, float8, float, int8, int, int8, int)       \
  IMPLEMENT_BODY_V (NAME, EXPR, 16, float16, float, int16, int, int16, int)   \
  __IF_FP64 (                                                                 \
      IMPLEMENT_BODY_V (NAME, EXPR, 1, double, double, int, int, long, long)  \
          IMPLEMENT_BODY_V (NAME, EXPR, 2, double2, double, int2, int, long2, \
                            long)                                             \
              IMPLEMENT_BODY_V (NAME, EXPR, 3, double3, double, int3, int,    \
                                long3, long)                                  \
                  IMPLEMENT_BODY_V (NAME, EXPR, 4, double4, double, int4,     \
                                    int, long4, long)                         \
                      IMPLEMENT_BODY_V (NAME, EXPR, 8, double8, double, int8, \
                                        int, long8, long)                     \
                          IMPLEMENT_BODY_V (NAME, EXPR, 16, double16, double, \
                                            int16, int, long16, long))

#define CALL_FUNC_V(NAME)                                                     \
  NAME##_float ();                                                            \
  NAME##_float2 ();                                                           \
  NAME##_float3 ();                                                           \
  NAME##_float4 ();                                                           \
  NAME##_float8 ();                                                           \
  NAME##_float16 ();                                                          \
  __IF_FP64 (NAME##_double (); NAME##_double2 (); NAME##_double3 ();          \
             NAME##_double4 (); NAME##_double8 (); NAME##_double16 ();)


#define IMPLEMENT_BODY_G(NAME, BODY, SIZE, GTYPE, SGTYPE, UGTYPE, SUGTYPE)  \
  void NAME##_##GTYPE()                                                     \
  {                                                                         \
    typedef GTYPE gtype;                                                    \
    typedef SGTYPE sgtype;                                                  \
    typedef UGTYPE ugtype;                                                  \
    typedef SUGTYPE sugtype;                                                \
    string const typename = #GTYPE;                                         \
    const int vecsize = SIZE;                                               \
    BODY;                                                                   \
  }
#define DEFINE_BODY_G(NAME, EXPR)                                           \
  IMPLEMENT_BODY_G(NAME, EXPR,  1, char    , char  , uchar   , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  2, char2   , char  , uchar2  , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  3, char3   , char  , uchar3  , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  4, char4   , char  , uchar4  , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  8, char8   , char  , uchar8  , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR, 16, char16  , char  , uchar16 , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  1, uchar   , uchar , uchar   , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  2, uchar2  , uchar , uchar2  , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  3, uchar3  , uchar , uchar3  , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  4, uchar4  , uchar , uchar4  , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  8, uchar8  , uchar , uchar8  , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR, 16, uchar16 , uchar , uchar16 , uchar )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  1, short   , short , ushort  , ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR,  2, short2  , short , ushort2 , ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR,  3, short3  , short , ushort3 , ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR,  4, short4  , short , ushort4 , ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR,  8, short8  , short , ushort8 , ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR, 16, short16 , short , ushort16, ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR,  1, ushort  , ushort, ushort  , ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR,  2, ushort2 , ushort, ushort2 , ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR,  3, ushort3 , ushort, ushort3 , ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR,  4, ushort4 , ushort, ushort4 , ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR,  8, ushort8 , ushort, ushort8 , ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR, 16, ushort16, ushort, ushort16, ushort)      \
  IMPLEMENT_BODY_G(NAME, EXPR,  1, int     , int   , uint    , uint  )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  2, int2    , int   , uint2   , uint  )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  3, int3    , int   , uint3   , uint  )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  4, int4    , int   , uint4   , uint  )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  8, int8    , int   , uint8   , uint  )      \
  IMPLEMENT_BODY_G(NAME, EXPR, 16, int16   , int   , uint16  , uint  )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  1, uint    , uint  , uint    , uint  )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  2, uint2   , uint  , uint2   , uint  )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  3, uint3   , uint  , uint3   , uint  )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  4, uint4   , uint  , uint4   , uint  )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  8, uint8   , uint  , uint8   , uint  )      \
  IMPLEMENT_BODY_G(NAME, EXPR, 16, uint16  , uint  , uint16  , uint  )      \
  __IF_INT64(                                                               \
  IMPLEMENT_BODY_G(NAME, EXPR,  1, long    , long  , ulong   , ulong )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  2, long2   , long  , ulong2  , ulong )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  3, long3   , long  , ulong3  , ulong )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  4, long4   , long  , ulong4  , ulong )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  8, long8   , long  , ulong8  , ulong )      \
  IMPLEMENT_BODY_G(NAME, EXPR, 16, long16  , long  , ulong16 , ulong )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  1, ulong   , ulong , ulong   , ulong )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  2, ulong2  , ulong , ulong2  , ulong )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  3, ulong3  , ulong , ulong3  , ulong )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  4, ulong4  , ulong , ulong4  , ulong )      \
  IMPLEMENT_BODY_G(NAME, EXPR,  8, ulong8  , ulong , ulong8  , ulong )      \
  IMPLEMENT_BODY_G(NAME, EXPR, 16, ulong16 , ulong , ulong16 , ulong ))

#define CALL_FUNC_G(NAME)                       \
  NAME##_char    ();                            \
  NAME##_char2   ();                            \
  NAME##_char3   ();                            \
  NAME##_char4   ();                            \
  NAME##_char8   ();                            \
  NAME##_char16  ();                            \
  NAME##_uchar   ();                            \
  NAME##_uchar2  ();                            \
  NAME##_uchar3  ();                            \
  NAME##_uchar4  ();                            \
  NAME##_uchar8  ();                            \
  NAME##_uchar16 ();                            \
  NAME##_short   ();                            \
  NAME##_short2  ();                            \
  NAME##_short3  ();                            \
  NAME##_short4  ();                            \
  NAME##_short8  ();                            \
  NAME##_short16 ();                            \
  NAME##_ushort  ();                            \
  NAME##_ushort2 ();                            \
  NAME##_ushort3 ();                            \
  NAME##_ushort4 ();                            \
  NAME##_ushort8 ();                            \
  NAME##_ushort16();                            \
  NAME##_int     ();                            \
  NAME##_int2    ();                            \
  NAME##_int3    ();                            \
  NAME##_int4    ();                            \
  NAME##_int8    ();                            \
  NAME##_int16   ();                            \
  NAME##_uint    ();                            \
  NAME##_uint2   ();                            \
  NAME##_uint3   ();                            \
  NAME##_uint4   ();                            \
  NAME##_uint8   ();                            \
  NAME##_uint16  ();                            \
  __IF_INT64(                                   \
  NAME##_long    ();                            \
  NAME##_long2   ();                            \
  NAME##_long3   ();                            \
  NAME##_long4   ();                            \
  NAME##_long8   ();                            \
  NAME##_long16  ();                            \
  NAME##_ulong   ();                            \
  NAME##_ulong2  ();                            \
  NAME##_ulong3  ();                            \
  NAME##_ulong4  ();                            \
  NAME##_ulong8  ();                            \
  NAME##_ulong16 ();)





#if __has_extension(c_generic_selections) && (__clang_major__ < 6)
#ifdef cl_khr_fp64
#define is_floating(T) _Generic((T)0, float : 1, double : 1, default : 0)
#else
#define is_floating(T) _Generic((T)0, float : 1, default : 0)
#endif
#else
#define is_floating(T) 1
#endif
#define is_signed(T) ((T)-1 < (T) + 1)
#define count_bits(T) (CHAR_BIT * sizeof (T))

#define ISNAN(x) (isnan (x) || as_int ((float)(x)) == as_int ((float)NAN))
#define ISEQ(x, y) (ISNAN (x) == ISNAN (y) && (ISNAN (x) || (x) == (y)))
