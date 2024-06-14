import matplotlib
matplotlib.use("Agg")


import matplotlib.pyplot as plt

#x_values = [128, 256, 512, 1024, 2048, 4096]
#y_values = [65, 257, 1025, 4097, 16385, 16384*4+1]

#x_values = [1, 2, 3, 4]
#y_values = [1.583, 1.11, 0.93, 0.84]

freeze_rmse_values = [0.02339809201657772, 0.028828689828515053, 0.038288380950689316, 0.04984824359416962, 0.06025683879852295, 0.06664843112230301, 0.07393575459718704, 0.08242091536521912, 0.08858775347471237, 0.09216747432947159, 0.09437235444784164, 0.09834787994623184, 0.1013442799448967, 0.10429652780294418, 0.10768933594226837, 0.11137279123067856, 0.11523927748203278, 0.1184099093079567, 0.12134160101413727, 0.12429428845643997, 0.1269693225622177, 0.12934401631355286, 0.13182590901851654, 0.13493095338344574, 0.13722273707389832, 0.13962450623512268, 0.1422262191772461, 0.14495554566383362, 0.14796960353851318, 0.1507202386856079]
base_rmse_values = [0.02444406785070896, 0.0415988452732563, 0.05575643479824066, 0.06857986003160477, 0.07519377022981644, 0.08379999548196793, 0.09070222079753876, 0.0967266857624054, 0.10196885466575623, 0.10982605814933777, 0.11693504452705383, 0.12184584140777588, 0.1262623518705368, 0.1301952749490738, 0.1335352510213852, 0.136525958776474, 0.1399158090353012, 0.14376991987228394, 0.1468440145254135, 0.14852285385131836, 0.14983372390270233, 0.15097597241401672, 0.15140393376350403, 0.15181663632392883, 0.1525305211544037, 0.15324391424655914, 0.1542699635028839, 0.15555071830749512, 0.15705733001232147, 0.1588476449251175]

x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]



plt.plot(x_values, freeze_rmse_values, label="encoder + afno frozen", color="blue")
plt.plot(x_values, base_rmse_values, label="afno base", color="red") 
#plt.yscale("log")
plt.xlabel("AR step")
plt.ylabel("rmse")
plt.title("rmse vs AR step")
plt.savefig("plot.png")

