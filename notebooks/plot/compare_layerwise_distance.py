# %%
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib as mpl
from matplotlib import rc
import seaborn as sns
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed


# fill in data
l3 = np.array([0.799202561378479, 0.6615535020828247, 0.6597975492477417, 0.6674417853355408, 0.8019217252731323, 0.9142443537712097])
l6 = np.array([0.62445148229599, 0.5815010666847229, 0.5812696218490601, 0.6335068941116333, 0.8428128957748413, 0.8444141149520874])

N = 6
ind = np.arange(N) * 14  # the x locations for the groups
width = 4.0      # the width of the bars
shift = 0.8


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, ax = plt.subplots(figsize=(8,6))
rects3 = ax.bar(ind + width * 2 + shift*2, l3, width, color='orange', ecolor='white')
rects6 = ax.bar(ind + width * 3 + shift*3, l6, width, color='royalblue', ecolor='white') # color='tomato', ecolor='k', hatch="x"

#ax.set_ylim([0.2, 400])
ax.set_xticks(ind + width  + shift + 8)
ax.set_xticklabels(["1", "2",  "3", "4",  "5", "6"], fontsize=44)
# plt.yticks(np.arange(0, 12500, 2000))
# ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# ax.yaxis.get_offset_text().set_fontsize(40)
plt.yticks(np.arange(0, 1.3, 0.1))
plt.ylim([0.55, 1.03])
ax.set_ylabel(r'$||\theta - \hat{\theta}^{(s)}||$', fontsize=44)

plt.tick_params(axis='x')


ax.legend(
    (rects3[0], rects6[0]),
    ( r'$\mathrm{Uniform~Projection}$', r'$\mathrm{Alg.~1}$',),
    loc=2, fontsize=38, frameon=False)

ax.yaxis.grid(True, lw=0.4)
ax.set_title(r'$\mathrm{Query~Layers}$', fontsize=44, x=0.483, y=1.02)
ax.tick_params(axis='both', which='major', labelsize=40)
ax.tick_params(axis='both', which='minor', labelsize=40)

plt.tight_layout()
plt.savefig('comparison_layerwise_distance_q.pdf', format='pdf', dpi=100)

# %%
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib as mpl
from matplotlib import rc
import seaborn as sns
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed


# fill in data
l3 = np.array([0.7921828031539917, 0.730654239654541, 0.6837455034255981, 0.7359063625335693, 0.9271995425224304, 0.972970187664032])
l6 = np.array([0.5869855284690857, 0.5923574566841125, 0.6452652215957642, 0.7048307657241821, 0.9182260632514954, 0.9609395265579224])
N = 6
ind = np.arange(N) * 14  # the x locations for the groups
width = 4.0      # the width of the bars
shift = 0.8



plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, ax = plt.subplots(figsize=(8,6))
rects3 = ax.bar(ind + width * 2 + shift*2, l3, width, color='orange', ecolor='white')
rects6 = ax.bar(ind + width * 3 + shift*3, l6, width, color='royalblue', ecolor='white') # color='tomato', ecolor='k', hatch="x"

#ax.set_ylim([0.2, 400])
ax.set_xticks(ind + width  + shift + 8)
ax.set_xticklabels(["1", "2",  "3", "4",  "5", "6"], fontsize=44)
# plt.yticks(np.arange(0, 12500, 2000))
# ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# ax.yaxis.get_offset_text().set_fontsize(40)
plt.yticks(np.arange(0, 1.3, 0.1))
plt.ylim([0.55, 1.03])
ax.set_ylabel(r'$||\theta - \hat{\theta}^{(s)}||$', fontsize=44)

plt.tick_params(axis='x')


# ax.legend(
#     (rects3[0], rects6[0]),
#     ( r'$\mathrm{Uniform~Projection}$', r'$\mathrm{Alg.~1}$',),
#     loc=2, fontsize=34, frameon=False)

ax.yaxis.grid(True, lw=0.4)
ax.set_title(r'$\mathrm{Key~Layers}$', fontsize=44, x=0.483, y=1.02)
ax.tick_params(axis='both', which='major', labelsize=40)
ax.tick_params(axis='both', which='minor', labelsize=40)

plt.tight_layout()
plt.savefig('comparison_layerwise_distance_k.pdf', format='pdf', dpi=100)

# %%
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib as mpl
from matplotlib import rc
import seaborn as sns
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed


# fill in data
l3 = np.array([0.5312597155570984, 0.5151911973953247, 0.5171226263046265, 0.5390536189079285, 0.6841398477554321, 0.6971816420555115])
l6 = np.array([0.4347642660140991, 0.4866939187049866, 0.5486354827880859, 0.6128819584846497, 0.7110241055488586, 0.8135297298431396])
N = 6
ind = np.arange(N) * 14  # the x locations for the groups
width = 4.0      # the width of the bars
shift = 0.8


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, ax = plt.subplots(figsize=(8,6))
rects3 = ax.bar(ind + width * 2 + shift*2, l3, width, color='orange', ecolor='white')
rects6 = ax.bar(ind + width * 3 + shift*3, l6, width, color='royalblue', ecolor='white') # color='tomato', ecolor='k', hatch="x"

#ax.set_ylim([0.2, 400])
#ax.set_ylabel(r'', fontsize=48)
ax.set_xticks(ind + width  + shift + 8)
ax.set_xticklabels(["1", "2",  "3", "4",  "5", "6"], fontsize=44)
# plt.yticks(np.arange(0, 12500, 2000))
# ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# ax.yaxis.get_offset_text().set_fontsize(40)
plt.yticks(np.arange(0, 1.3, 0.1))
plt.ylim([0.37, .84])
ax.set_ylabel(r'$||\theta - \hat{\theta}^{(s)}||$', fontsize=44)

plt.tick_params(axis='x')


# ax.legend(
#     (rects3[0], rects6[0]),
#     ( r'$\mathrm{Uniform~Projection}$', r'$\mathrm{Alg.~1}$',),
#     loc=2, fontsize=28, frameon=False)


ax.yaxis.grid(True, lw=0.4)
ax.set_title(r'$\mathrm{Value~Layers}$', fontsize=44, x=0.483, y=1.02)
ax.tick_params(axis='both', which='major', labelsize=40)
ax.tick_params(axis='both', which='minor', labelsize=40)

plt.tight_layout()
plt.savefig('comparison_layerwise_distance_v.pdf', format='pdf', dpi=100)

# %%
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib as mpl
from matplotlib import rc
import seaborn as sns
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed


# fill in data
l3 = np.array([0.7110385894775391, 0.5461869239807129, 0.6069779992103577, 0.699951171875, 0.8638054132461548, 0.8401901125907898])
l6 = np.array([0.6403015661239624, 0.5075825476646423, 0.6469258069992065, 0.7732915282249451, 0.9163628220558167, 0.9731821417808533])
N = 6
ind = np.arange(N) * 14  # the x locations for the groups
width = 4.0      # the width of the bars
shift = 0.8


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, ax = plt.subplots(figsize=(8,6))
rects3 = ax.bar(ind + width * 2 + shift*2, l3, width, color='orange', ecolor='white')
rects6 = ax.bar(ind + width * 3 + shift*3, l6, width, color='royalblue', ecolor='white') # color='tomato', ecolor='k', hatch="x"

#ax.set_ylim([0.2, 400])
#ax.set_ylabel(r'', fontsize=48)
ax.set_xticks(ind + width  + shift + 8)
ax.set_xticklabels(["1", "2",  "3", "4",  "5", "6"], fontsize=40)
# plt.yticks(np.arange(0, 12500, 2000))
# ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# ax.yaxis.get_offset_text().set_fontsize(40)
plt.yticks(np.arange(0, 1.3, 0.1))
plt.ylim([0.47, 1.04])
ax.set_ylabel(r'$||\theta - \hat{\theta}^{(s)}||$', fontsize=40)

plt.tick_params(axis='x')


ax.legend(
    (rects3[0], rects6[0]),
    ( r'$\mathrm{Uniform~Projection}$', r'$\mathrm{Alg.~1}$',),
    loc=2, fontsize=32, frameon=False)


ax.yaxis.grid(True, lw=0.4)
ax.set_title(r'$\mathrm{Output~Layers}$', fontsize=40, x=0.483, y=1.02)
ax.tick_params(axis='both', which='major', labelsize=40)
ax.tick_params(axis='both', which='minor', labelsize=40)

plt.tight_layout()
plt.savefig('comparison_layerwise_distance_o.pdf', format='pdf', dpi=100)


# %%
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib as mpl
from matplotlib import rc
import seaborn as sns
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed


# fill in data
l3 = np.array([1.3993558883666992, 1.148517370223999, 1.0761348009109497, 1.1157331466674805, 1.2097067832946777, 1.347928762435913])
l6 = np.array([1.1145967245101929, 1.1297479104995728, 1.1748976707458496, 1.2133122682571411, 1.2767486572265625, 1.3839542865753174])
N = 6
ind = np.arange(N) * 14  # the x locations for the groups
width = 4.0      # the width of the bars
shift = 0.8


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, ax = plt.subplots(figsize=(8,6))
rects3 = ax.bar(ind + width * 2 + shift*2, l3, width, color='orange', ecolor='white')
rects6 = ax.bar(ind + width * 3 + shift*3, l6, width, color='royalblue', ecolor='white') # color='tomato', ecolor='k', hatch="x"

#ax.set_ylim([0.2, 400])
#ax.set_ylabel(r'', fontsize=48)
ax.set_xticks(ind + width  + shift + 8)
ax.set_xticklabels(["1", "2",  "3", "4",  "5", "6"], fontsize=40)
# plt.yticks(np.arange(0, 12500, 2000))
# ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# ax.yaxis.get_offset_text().set_fontsize(40)
plt.yticks(np.arange(0, 1.8, 0.1))
plt.ylim([0.97, 1.54])
ax.set_ylabel(r'$||\theta - \hat{\theta}^{(s)}||$', fontsize=40)

plt.tick_params(axis='x')


# ax.legend(
#     (rects3[0], rects6[0]),
#     ( r'$\mathrm{Uniform~Projection}$', r'$\mathrm{Alg.~1}$',),
#     loc=2, fontsize=32, frameon=False)


ax.yaxis.grid(True, lw=0.4)
ax.set_title(r'$\mathrm{First~FFN~Layers}$', fontsize=40, x=0.483, y=1.02)
ax.tick_params(axis='both', which='major', labelsize=40)
ax.tick_params(axis='both', which='minor', labelsize=40)

plt.tight_layout()
plt.savefig('comparison_layerwise_distance_f1.pdf', format='pdf', dpi=100)

# %%
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib as mpl
from matplotlib import rc
import seaborn as sns
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed


# fill in data
l3 = np.array([1.189350962638855, 1.011918067932129, 1.009034276008606, 1.0701971054077148, 1.1593302488327026, 1.2658885717391968])
l6 = np.array([.9863807678222656, 1.0302900695800781, 1.0892558097839355, 1.1489094495773315, 1.2178889513015747, 1.3086481094360352])
N = 6
ind = np.arange(N) * 14  # the x locations for the groups
width = 4.0      # the width of the bars
shift = 0.8


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, ax = plt.subplots(figsize=(8,6))
rects3 = ax.bar(ind + width * 2 + shift*2, l3, width, color='orange', ecolor='white')
rects6 = ax.bar(ind + width * 3 + shift*3, l6, width, color='royalblue', ecolor='white') # color='tomato', ecolor='k', hatch="x"

#ax.set_ylim([0.2, 400])
#ax.set_ylabel(r'', fontsize=48)
ax.set_xticks(ind + width  + shift + 8)
ax.set_xticklabels(["1", "2",  "3", "4",  "5", "6"], fontsize=40)
# plt.yticks(np.arange(0, 12500, 2000))
# ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# ax.yaxis.get_offset_text().set_fontsize(40)
plt.yticks(np.arange(0, 1.8, 0.1))
plt.ylim([0.87, 1.44])
ax.set_ylabel(r'$||\theta - \hat{\theta}^{(s)}||$', fontsize=40)

plt.tick_params(axis='x')


# ax.legend(
#     (rects3[0], rects6[0]),
#     ( r'$\mathrm{Uniform~Projection}$', r'$\mathrm{Alg.~1}$',),
#     loc=2, fontsize=32, frameon=False)


ax.yaxis.grid(True, lw=0.4)
ax.set_title(r'$\mathrm{Second~FFN~Layers}$', fontsize=40, x=0.483, y=1.02)
ax.tick_params(axis='both', which='major', labelsize=40)
ax.tick_params(axis='both', which='minor', labelsize=40)

plt.tight_layout()
plt.savefig('comparison_layerwise_distance_f2.pdf', format='pdf', dpi=100)




# %%
# q [0.6325408220291138, 0.6225249767303467, 0.6422157883644104, 0.7422738075256348, 0.9326025247573853, 0.9981677532196045]
# k [0.5869855284690857, 0.5923574566841125, 0.6452652215957642, 0.7048307657241821, 0.9182260632514954, 0.9609395265579224]
# v [0.4347642660140991, 0.4866939187049866, 0.5486354827880859, 0.6128819584846497, 0.7110241055488586, 0.8135297298431396]
# o [0.6603015661239624, 0.5675825476646423, 0.6469258069992065, 0.7732915282249451, 0.9163628220558167, 0.9731821417808533]
# c_fc [1.1145967245101929, 1.2097479104995728, 1.1748976707458496, 1.2133122682571411, 1.2767486572265625, 1.3839542865753174]
# c_proj [1.0363807678222656, 1.0902900695800781, 1.0892558097839355, 1.1489094495773315, 1.2178889513015747, 1.3086481094360352]

# q [0.8074303865432739, 0.7206326127052307, 0.6946344375610352, 0.7867770791053772, 0.9753321409225464, 1.0563876628875732]
# k [0.7921828031539917, 0.730654239654541, 0.6837455034255981, 0.7359063625335693, 0.9271995425224304, 0.972970187664032]
# v [0.5312597155570984, 0.5151911973953247, 0.5171226263046265, 0.5390536189079285, 0.6841398477554321, 0.6971816420555115]
# o [0.7110385894775391, 0.5461869239807129, 0.6069779992103577, 0.699951171875, 0.8638054132461548, 0.8401901125907898]
# c_fc [1.3993558883666992, 1.148517370223999, 1.0761348009109497, 1.1157331466674805, 1.2097067832946777, 1.347928762435913]
# c_proj [1.189350962638855, 1.011918067932129, 1.009034276008606, 1.0701971054077148, 1.1593302488327026, 1.2658885717391968]