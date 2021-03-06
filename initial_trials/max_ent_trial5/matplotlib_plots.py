import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
# from sklearn.preprocessing import normalize


trajectories_z0 = [[0, 0, 0, 0] for i in range(1331)]
default_points = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
i = 0
for z in default_points:
  for y in default_points:
    for x in default_points:
      trajectories_z0[i] = [x, y, z, 0.]
      i += 1

# weights = np.array(
#   [
#     0.90759588,
#     4.77365172,
#     0.8771005,
#     0.43324404,
#     6.33486176,
#     0.87008563,
#     0.60639639,
#     4.76273361,
#     0.17210085,
#     0.02747048,
#     1.08775725,
#     0.46268714,
#     0.39770698,
#     0.42458757,
#     0.4958771,
#     0.82629336,
#     4.68860071,
#     0.20892782,
#     0.77898644,
#     3.32710488,
#     0.22604133,
#     0.7976902,
#     0.7246625,
#     0.60428786,
#     0.78655751,
#     4.74956047,
#     0.20839334
#   ]
# )

weights = np.array([ 0.86606,    0.83231,    0.36573,    0.88065,    0.43141,    8.09653,    0.48275,    0.63349,
                     0.22278,    0.65300,    0.39058,    0.70320,    0.45969,    0.50013,    0.89983,    0.48156,
                     8.45762,    0.17549,    0.17827,    0.93676,    0.35098,    0.32827,    0.89523,    0.91574,
                     0.38595,    0.82771,    0.84156,    8.15301,    0.13228,    0.80403,    0.00957,    0.83780,
                     0.86237,    0.78184,    0.12404,    0.52459,    0.98938,    0.46508,    8.04134,    0.26305,
                     0.49731,    0.28017,    0.27976,    0.20002,    0.46367,    0.54741,    0.93018,    0.66636,
                     0.42815,    8.57352,    0.08831,    0.32845,    0.97826,    0.76923,    0.30592,    0.60836,
                     0.55568,    0.88519,    0.94008,    0.87352,    8.77289,    0.08207,    0.36324,    0.40164,
                     0.05353,    0.34603,    0.45539,    0.38059,    0.65810,    0.73721,    0.36884,    8.58656,
                     0.29147,    0.00275,    0.40088,    0.55660,    0.86510,    0.02823,    0.87297,    0.47724,
                     0.63697,    0.84070,    8.56395,    0.89196,    0.94764,    0.52526,    0.92291,    0.17491,
                     0.11700,    0.11084,    0.66124,    0.07273,    0.94089,    8.53461,    0.89326,    0.36706,
                     0.17802,    0.53724,    0.96983,    0.56480,    0.20451,    0.33316,    0.69080,    0.63368,
                     8.43536,    0.61112,    0.74471,    0.78599,    0.18430,    0.58020,    0.18954,    0.60108,
                     0.94191,    0.26950,    0.89086,    8.73328,    0.02798,    0.11270,    0.16027,    0.13783,
                     0.05105,    0.85937,    0.80662,    0.59525,    0.74255,    0.89817,    8.09508,    0.12638,
                     0.21415,    0.96650,    0.12495,    0.88526,    0.09146,    0.42899,    0.70579,    0.25391,
                     0.41425,    0.11668,    0.07310,    0.23312,    0.50665,    0.10344,    0.75825,    0.71842,
                     0.40381,    0.18298,    0.36077,    0.26998,    0.92945,    0.23136,    0.59756,    0.98148,
                     0.18478,    0.35717,    0.14017,    0.74673,    0.45086,    0.09237,    0.20714,    0.71585,
                     0.49646,    0.24563,    0.99324,    0.45483,    0.29137,    0.31014,    0.75493,    0.91689,
                     0.95660,    0.84002,    0.18656,    0.62444,    0.20679,    0.62989,    0.14484,    0.03144,
                     0.41225,    0.91965,    0.91157,    0.45958,    0.84885,    0.08504,    0.76278,    0.32672,
                     0.39368,    0.79983,    0.10128,    0.68598,    0.83559,    0.21271,    0.33390,    0.11628,
                     0.71137,    0.65598,    0.78404,    0.47902,    0.26255,    0.77631,    0.04882,    0.62668,
                     0.34303,    0.52945,    0.80601,    0.92397,    0.14185,    0.89462,    0.01004,    0.15316,
                     0.02876,    0.64054,    0.15996,    0.40866,    0.94214,    0.31214,    0.14124,    0.57567,
                     0.67908,    0.43659,    0.52660,    0.61059,    0.66739,    0.60736,    0.42181,    0.06852,
                     0.30142,    0.30690,    0.30800,    0.45645,    0.73653,    0.64136,    0.92888,    0.48359,
                     0.45741,    0.23591,    0.07896,    0.34559,   13.91303,    0.56566,    0.17381,    0.98895,
                     0.28075,    0.22149,    0.21180,    0.27492,    0.24944,    0.77786,    0.50302,    8.25382,
                     0.50928,    0.22889,    0.26443,    0.81160,    0.62592,    0.56823,    0.37586,    0.06454,
                     0.19488,    0.31179,    0.15198,    0.81571,    0.79824,    0.02611,    0.21241,    0.98750,
                     0.69514,    0.26780,    0.56631,    0.98667,    0.59943,    0.00309,    0.84204,    0.21672,
                     0.93846,    0.90865,    0.01556,    0.45905,    0.34803,    0.32157,    0.94343,    0.62096,
                     0.49086,    0.92539,    0.60237,    0.12542,    0.78568,    0.70110,    0.51186,    0.72851,
                     0.85708,    0.34506,    0.22070,    0.00386,    0.08566,    0.59610,    0.97948,    0.56980,
                     0.76479,    0.55984,    0.84063,    0.46915,    0.48015,    0.35191,    0.73344,    0.16073,
                     0.18566,    0.80631,    0.99631,    0.17507,    0.78418,    0.67018,    0.54158,    0.59866,
                     0.29096,    0.52003,    0.16887,    0.80251,    0.04077,    0.21190,    0.65894,    0.71938,
                     0.15865,    0.45670,    0.39790,    0.88212,    0.81452,    0.45652,    0.82504,    0.80774,
                     0.77313,    0.77773,    0.88256,    0.41025,    0.28867,    0.34217,    0.10101,    0.65445,
                     0.64544,    0.44004,    0.85440,    0.93771,    0.51563,    0.77434,    0.06947,    0.55451,
                     0.56954,    0.51177,    0.81949,    0.29058,    0.02071,    0.23019,    0.10884,    0.21713,
                     0.28555,    0.68137,    0.06308,    0.78964,    0.90662,    8.48588,    0.13338,    0.97124,
                     0.28128,    0.91960,    0.78790,    0.67392,    0.67800,    0.63724,    0.83277,    0.18829,
                     11.55437,    0.49343,    0.02084,    0.98292,    0.04159,    0.49970,    0.92490,    0.83001,
                     0.37967,    0.60845,    0.77930,    0.12143,    0.95024,    0.52538,    0.30988,    0.17887,
                     0.04590,    0.70208,    0.88555,    0.04735,    0.97170,    0.87269,    0.30152,    0.71948,
                     0.43896,    0.54505,    0.49488,    0.59656,    0.10639,    0.99643,    0.13154,    0.20760,
                     0.24538,    0.42594,    0.60919,    0.32661,    0.13909,    0.66058,    0.76267,    0.12587,
                     0.74871,    0.27353,    0.06857,    0.66941,    0.68933,    0.40911,    0.48100,    0.40839,
                     0.29233,    0.01798,    0.45945,    0.82710,    0.84453,    0.74455,    0.82811,    0.90751,
                     0.72218,    0.22907,    0.23105,    0.29386,    0.03299,    0.07297,    0.65535,    0.39458,
                     0.18482,    0.10684,    0.69006,    0.03536,    0.77620,    0.07033,    0.85254,    0.44963,
                     0.44249,    0.42462,    0.72545,    0.22172,    0.54252,    0.99930,    0.27761,    0.58339,
                     0.37403,    0.12483,    0.66204,    0.95506,    0.26642,    0.07111,    0.75879,    0.55934,
                     0.27706,    0.17421,    0.63632,    0.32397,    0.62688,    0.00929,    0.33517,    0.47173,
                     0.19735,    0.25440,    0.23137,    0.13805,    0.29358,    0.57446,    0.16891,    0.17486,
                     0.81168,    0.56355,    0.19217,    0.77762,    0.43355,    0.51899,    8.74231,    0.04208,
                     0.63581,    0.72404,    0.20752,    0.67161,    0.49087,    0.66678,    0.53526,    0.93921,
                     0.14829,    8.39485,    0.76900,    0.38939,    0.30797,    0.69380,    0.93981,    0.64036,
                     0.37906,    0.54725,    0.79273,    0.92407,    0.46334,    0.15352,    0.81080,    0.31234,
                     0.84989,    0.91849,    0.28454,    0.07828,    0.44564,    0.61013,    0.27826,    0.79339,
                     0.35125,    0.34566,    0.37643,    0.20707,    0.75552,    0.43252,    0.67397,    0.41246,
                     0.36061,    0.77067,    0.82372,    0.41645,    0.38525,    0.09377,    0.92424,    0.01445,
                     0.24964,    0.30950,    0.43543,    0.93096,    0.56174,    0.51897,    0.45165,    0.61233,
                     0.62962,    0.23426,    0.23870,    0.47391,    0.08279,    0.70883,    0.99494,    0.33809,
                     0.19212,    0.59100,    0.26717,    0.56835,    0.32082,    0.51294,    0.65807,    0.48010,
                     0.31026,    0.40197,    0.43840,    0.89893,    0.16454,    0.31181,    0.13842,    0.81631,
                     0.89824,    0.73875,    0.55308,    0.16719,    0.35501,    0.49828,    0.15169,    0.70433,
                     0.51742,    0.34092,    0.63148,    0.66503,    0.65517,    0.95970,    0.30421,    0.01801,
                     0.37439,    0.18743,    0.29988,    0.60631,    0.95896,    0.34174,    0.99733,    0.39911,
                     0.09226,    0.45185,    0.89540,    0.34566,    0.55707,    0.71923,    0.19182,    0.49286,
                     0.33445,    0.07131,    0.42297,    0.43129,    0.79597,    0.01133,    0.56211,    8.72557,
                     0.59397,    0.96041,    0.35297,    0.80701,    0.02397,    0.84990,    0.55624,    0.88276,
                     0.58454,    0.72116,    8.29342,    0.08053,    0.68322,    0.89053,    0.15748,    0.11644,
                     0.74621,    0.03926,    0.87751,    0.05730,    0.42855,    0.77222,    0.62937,    0.84752,
                     0.72300,    0.70865,    0.57909,    0.11975,    0.26066,    0.32511,    0.61659,    0.23415,
                     0.02240,    0.83902,    0.88202,    0.58408,    0.12551,    0.61981,    0.28566,    0.15060,
                     0.64480,    0.94000,    0.78554,    0.28172,    0.83536,    0.97732,    0.77693,    0.76901,
                     0.71192,    0.17845,    0.68893,    0.08023,    0.31377,    0.84967,    0.46194,    0.08320,
                     0.45239,    0.14372,    0.47152,    0.89625,    0.30673,    0.83032,    0.52956,    0.83302,
                     0.51186,    0.32895,    0.76231,    0.68038,    0.36668,    0.06482,    0.58357,    0.86842,
                     0.02698,    0.95348,    0.93468,    0.88380,    0.08353,    0.97158,    0.22825,    0.84633,
                     0.69256,    0.27166,    0.63432,    0.39849,    0.81319,    0.95714,    0.61430,    0.68423,
                     0.78645,    0.46231,    0.78052,    0.82758,    0.51594,    0.09906,    0.99372,    0.96676,
                     0.54478,    0.38459,    0.22727,    0.20587,    0.44240,    0.36964,    0.90205,    0.17188,
                     0.55187,    0.99610,    0.21843,    0.78132,    0.97636,    0.88978,    0.28759,    0.51406,
                     0.05120,    0.26030,    0.00131,    0.78410,    0.60316,    0.16648,    0.15094,    0.43473,
                     8.78875,    0.52753,    0.46545,    0.95013,    0.35292,    0.61687,    0.71009,    0.78465,
                     0.23820,    0.29654,    0.83010,    8.09434,    0.24569,    0.53284,    0.64908,    0.10107,
                     0.43242,    0.02169,    0.21706,    0.33148,    0.77758,    0.73083,    0.31437,    0.54423,
                     0.30026,    0.90927,    0.60095,    0.60840,    0.86495,    0.46947,    0.37314,    0.89167,
                     0.43115,    0.34043,    0.51235,    0.93971,    0.34013,    0.77146,    0.12911,    0.18998,
                     0.16081,    0.70305,    0.77657,    0.59798,    0.36222,    0.75020,    0.10372,    0.47000,
                     0.44457,    0.05798,    0.68049,    0.74227,    0.36331,    0.70801,    0.19057,    0.04736,
                     0.14542,    0.23696,    0.38687,    0.79483,    0.49170,    0.45183,    0.22816,    0.83092,
                     0.59064,    0.87170,    0.59560,    0.21333,    0.97443,    0.24253,    0.68951,    0.34349,
                     0.17543,    0.02265,    0.50499,    0.06743,    0.73950,    0.98350,    0.10978,    0.55272,
                     0.81583,    0.75593,    0.81751,    0.42089,    0.53847,    0.10709,    0.05547,    0.75561,
                     0.48261,    0.73225,    0.03328,    0.83095,    0.09975,    0.26411,    0.65159,    0.11538,
                     0.54161,    0.09971,    0.24840,    0.34376,    0.99199,    0.22528,    0.58800,    0.89147,
                     0.48139,    0.45749,    0.25095,    0.17768,    0.66404,    0.04633,    0.45263,    0.22793,
                     0.62404,    0.33014,    0.71408,    0.35605,    0.50394,    0.48741,    0.75575,    0.95428,
                     0.66236,    0.79536,    0.00319,    0.12524,    0.15554,    0.38994,    0.01801,    0.46796,
                     0.52084,    0.71673,    0.06421,    0.80492,    8.36966,    0.91589,    0.54096,    0.34604,
                     0.61032,    0.98681,    0.50548,    0.99665,    0.44434,    0.79609,    0.35812,    0.87599,
                     0.50017,    0.29037,    0.08794,    0.02938,    0.72119,    0.13936,    0.94934,    0.89772,
                     0.22071,    0.79370,    0.37970,    0.79350,    0.63759,    0.01817,    0.92606,    0.71444,
                     0.68657,    0.91361,    0.11857,    0.14572,    0.20615,    0.04446,    0.76434,    0.19979,
                     0.96606,    0.03424,    0.53181,    0.03459,    0.01364,    0.92605,    0.75164,    0.82889,
                     0.96646,    0.59091,    0.15745,    0.27066,    0.51219,    0.41432,    0.13094,    0.51988,
                     0.81029,    0.60508,    0.27121,    0.71210,    0.76494,    0.15917,    0.62312,    0.41200,
                     0.45672,    0.31942,    0.53526,    0.24350,    0.89689,    0.60864,    0.56976,    0.58488,
                     0.12329,    0.84833,    0.46295,    0.62916,    0.70355,    0.78838,    0.24800,    0.40638,
                     0.82993,    0.98059,    0.16235,    0.53101,    0.86464,    0.23273,    0.67726,    0.02169,
                     0.41828,    0.69413,    0.72323,    0.89867,    0.00666,    0.20075,    0.42341,    0.06545,
                     0.99140,    0.90083,    0.47994,    0.74666,    0.36867,    0.14330,    0.93605,    0.19744,
                     0.45004,    0.72188,    0.16163,    0.59613,    0.05618,    0.15809,    0.65005,    0.38586,
                     0.35247,    0.92090,    8.85157,    0.52833,    0.66790,    0.38261,    0.27341,    0.74894,
                     0.80119,    0.43474,    0.26773,    0.09087,    0.55951,    8.81902,    0.67411,    0.15127,
                     0.81321,    0.99818,    0.47361,    0.05732,    0.05440,    0.23546,    0.91842,    0.96853,
                     0.41251,    0.30757,    0.72286,    0.32235,    0.43596,    0.16056,    0.13231,    0.73978,
                     0.14538,    0.51979,    0.79940,    0.66000,    0.29996,    0.74033,    0.42738,    0.76141,
                     0.89570,    0.92081,    0.04213,    0.56995,    0.80538,    0.19339,    0.94511,    0.80837,
                     0.25371,    0.05057,    0.15977,    0.06955,    0.00970,    0.01809,    0.30737,    0.95288,
                     0.86361,    0.35174,    0.60050,    0.90623,    0.47128,    0.37581,    0.56422,    0.63380,
                     0.10621,    0.43998,    0.43633,    0.37703,    0.15937,    0.03456,    0.92829,    0.98259,
                     0.86450,    0.07983,    0.70096,    0.72882,    0.06969,    0.79619,    0.15399,    0.81445,
                     0.22955,    0.64109,    0.52913,    0.04403,    0.11168,    0.43959,    0.94190,    0.32419,
                     0.70461,    0.56591,    0.83883,    0.15249,    0.37261,    0.99893,    0.57496,    0.73703,
                     0.05708,    0.86895,    0.53691,    0.78671,    0.47400,    0.45574,    0.81661,    0.65510,
                     0.06249,    0.75644,    0.94035,    0.24743,    0.58775,    0.08179,    0.20769,    0.40918,
                     0.05149,    0.59233,    0.38418,    0.94767,    0.40522,    0.06346,    0.79694,    0.38958,
                     0.44968,    0.45492,    0.94359,    8.85374,    0.41015,    0.74741,    0.41969,    0.81110,
                     0.79946,    0.49500,    0.09164,    0.88478,    0.47842,    0.67052,    0.82374,    0.10576,
                     0.43968,    0.22823,    0.76403,    0.95874,    0.32679,    0.87107,    0.58244,    0.14863,
                     0.69722,    0.62853,   -2.00515,    0.52854,    0.02848,    0.63980,    0.60695,    0.48047,
                     0.44925,    0.00939,    0.04707,    0.36945,    0.04491,    0.80156,    0.14096,    0.45283,
                     0.39572,    0.14768,    0.48269,    0.57304,    0.77467,    0.65643,    0.57472,    0.05631,
                     0.18792,    0.45527,    0.57712,    0.77502,    0.76985,    0.43767,    0.92996,    0.47475,
                     0.25656,    0.70372,    0.58827,    0.59044,    0.37880,    0.18544,    0.04020,    0.68618,
                     0.80915,    0.25466,    0.05850,    0.75434,    0.01398,    0.28469,    0.37772,    0.41109,
                     0.49796,    0.02678,    0.35879,    0.76657,    0.87208,    0.74961,    0.11721,    0.73542,
                     0.51364,    0.59284,    0.62132,    0.56001,    0.38204,    0.06201,    0.84133,    0.30888,
                     0.63000,    0.85759,    0.57382,    0.04057,    0.60318,    0.78979,    0.35866,    0.69027,
                     0.20584,    0.94213,    0.67975,    0.94983,    0.32891,    0.54046,    0.54292,    0.09238,
                     0.21977,    0.59055,    0.81570,    0.39318,    0.57223,    0.62051,    0.58039,    0.53808,
                     0.64988,    0.62057,    0.57613,    0.62425,    0.11958,    0.43870,    0.47647,    0.12566,
                     0.58883,    0.03753,    0.35327,    0.54154,    8.96078,    0.26898,    0.09697,    0.88066,
                     0.83887,    0.00879,    0.73489,    0.14313,    0.91515,    0.92804,    0.57323,    5.36694,
                     0.73675,    0.37697,    0.49263,    0.86585,    0.79449,    0.53104,    0.57565,    0.03924,
                     0.19305,    0.41072,    0.31136,    0.68538,    0.05464,    0.57504,    0.18413,    0.41380,
                     0.39582,    0.44707,    0.79045,    0.83497,    0.37627,    0.58076,    0.80799,    0.28740,
                     0.09940,    0.37530,    0.48598,    0.80533,    0.52863,    0.38378,    0.64624,    0.97195,
                     0.41401,    0.45348,    0.29574,    0.98287,    0.11765,    0.21454,    0.88715,    0.28245,
                     0.58811,    0.31512,    0.70104,    0.21242,    0.96732,    0.76459,    0.18028,    0.03213,
                     0.66222,    0.71905,    0.82666,    0.92921,    0.41742,    0.30730,    0.39884,    0.94566,
                     0.73405,    0.92883,    0.87678,    0.38588,    0.73689,    0.74803,    0.36897,    0.44296,
                     0.57116,    0.03314,    0.25149,    0.11172,    0.35919,    0.66736,    0.54074,    0.44695,
                     0.62746,    0.98569,    0.29430,    0.41188,    0.02561,    0.53059,    0.00311,    0.36874,
                     0.05705,    0.90674,    0.02452,    0.77775,    0.35716,    0.98764,    0.99478,    0.76482,
                     0.59376,    0.17029,    0.10382,    0.79658,    0.00892,    0.28215,    0.14567,    0.18449,
                     0.03274,    0.63881,    0.65990,    0.89727,    0.43747,    0.67325,    0.75189,    0.57326,
                     0.56659,    0.18759,    0.06218,    0.01805,    0.88160,   10.83546,    0.28012,    0.50686,
                     0.82691,    0.05311,    0.43345 ])


Mat = weights.reshape((11, 11, 11), order='F')
mFlat = Mat.flatten()
print("mflat is ", mFlat)
norm = mFlat/np.linalg.norm(mFlat)
# print("norm is ", norm)
# norm_thresh = np.zeros(1331)
# for i in range(len(norm)):
#   if norm[i] >=0.2:
#     norm_thresh[i] = 1
#   else:
#     norm_thresh[i] = 0.1
#
# print("norm thresh is ", norm_thresh)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(1331):
  # color = 'red'
  # label = 'z = -0.5'
  # if i >= 9 and i < 18:
  #   color = 'blue'
  #   label = 'z = 0'
  # elif i >= 18:
  #   color = 'green'
  #   label = 'z = 0.5'
    # print(i)
  trajectories_z0[i][3] = norm[i]
  ax.scatter(
    trajectories_z0[i][0],
    trajectories_z0[i][1],
    trajectories_z0[i][2],
    s=trajectories_z0[i][3] * 100,
    marker="o"
  )
ax.legend(loc=8, framealpha=1, fontsize=8)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
plt.title('Distribution of Reward Function for a custom trajectory')
fig.savefig("/home/vignesh/Desktop/grid11_3d_plot")