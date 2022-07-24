import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
#from matplotlib import font_manager, rc

#functions
def q_star(c_i, c_bar, mu, D, beta_range, n):
    
    q_list = []
    
    for i in beta_range:
        
        q_i = i*((c_bar - ((n-(1/i))/n)*c_i)/(c_bar))*pow(((mu*D*(((n-(1/i))/n)))/(c_bar)),i)
        q_list.append(q_i)
    
    return q_list

#input parameters
n = 4

st.sidebar.write("Please insert firms' marginal cost following the condition:\n")
st.sidebar.latex(r'''\textrm{(1) } c_4 < \frac{6}{5}(\frac{c_1 + c_2 + c_3}{3})''')
st.sidebar.latex(r'''\textrm{(2) } c_3 < \frac{4}{3}(\frac{c_1 + c_2}{2})''')

c_list = list(np.arange(n))
name_list = list(np.arange(n))

c_list[0] = st.sidebar.number_input('Insert firm 1\'s marginal cost (c1)', value=1.0, step=0.01)
name_list[0] = "firm 1 (c1)"
#st.write('The current number is ', c_list[0])

c_list[1] = st.sidebar.number_input('Insert firm 2\'s marginal cost (c2)', min_value = c_list[0], step=0.01)
name_list[1] = "firm 2 (c2)"
c_list[2] = st.sidebar.number_input('Insert firm 3\'s marginal cost (c3)', min_value = c_list[1], step=0.01)
name_list[2] = "firm 3 (c3)"
c_list[3] = st.sidebar.number_input('Insert firm 4\'s marginal cost (c4)', min_value = c_list[2], step=0.01)
name_list[3] = "firm 4 (c4)"

c_bar4 = sum(c_list)/len(c_list)
c_bar3 = (c_list[0]+c_list[1]+c_list[2])/3.

mu = st.sidebar.slider('Insert the market size parameter (mu)',
                           1.0, 20.0, 8.624)

D = 1

if (c_list[3]-c_bar4)==0:
    beta_max = 1
else:
    beta_max = (c_list[3]/(c_list[3]-c_bar4))*(1/n)

#beta_range = np.arange(0.25, beta_max, 0.01)
beta_range = np.arange(2./3.+0.0001, 2.0-0.0001, 0.01)

plot_q = pd.DataFrame(beta_range, columns=['beta'])
    
for i in range(0, n):
        
    values = q_star(c_list[i], c_bar4, mu, D, beta_range, n)
    label = 'q'+str(i+1)
    plot_q[label] = values

for i in range(0, 3):
    
    values = q_star(c_list[i], c_bar3, mu, D, beta_range, 3)
    label = 'a'+str(i+1)
    plot_q[label] = values
    
plot_q['Q'] = plot_q['q1']+plot_q['q2']+plot_q['q3']+plot_q['q4']
plot_q['P'] = c_bar4*n/(n-plot_q['beta'])
plot_q['p1'] = (plot_q['P']-c_list[0])*plot_q['q1']
plot_q['p2'] = (plot_q['P']-c_list[1])*plot_q['q2']
plot_q['p3'] = (plot_q['P']-c_list[2])*plot_q['q3']
plot_q['p4'] = (plot_q['P']-c_list[3])*plot_q['q4']
plot_q['s1'] = plot_q['q1'] / plot_q['Q']
plot_q['s2'] = plot_q['q2'] / plot_q['Q']
plot_q['s3'] = plot_q['q3'] / plot_q['Q']
plot_q['s4'] = plot_q['q4'] / plot_q['Q']
plot_q['p11'] = plot_q['p1']/plot_q['p1']
plot_q['p21'] = plot_q['p2']/plot_q['p1']
plot_q['p31'] = plot_q['p3']/plot_q['p1']
plot_q['p41'] = plot_q['p4']/plot_q['p1']
plot_q['zero'] = 0

plot_q['A'] = plot_q['a1']+plot_q['a2']+plot_q['a3']
plot_q['B'] = c_bar3*3/(3-plot_q['beta'])
plot_q['b1'] = (plot_q['B']-c_list[0])*plot_q['a1']
plot_q['b2'] = (plot_q['B']-c_list[1])*plot_q['a2']
plot_q['b3'] = (plot_q['B']-c_list[2])*plot_q['a3']
plot_q['c1'] = plot_q['a1'] / plot_q['A']
plot_q['c2'] = plot_q['a2'] / plot_q['A']
plot_q['c3'] = plot_q['a3'] / plot_q['A']
plot_q['d1'] = plot_q['b1'] / plot_q['b1']
plot_q['d2'] = plot_q['b2'] / plot_q['b1']
plot_q['d3'] = plot_q['b3'] / plot_q['b1']
plot_q['a1q1'] = plot_q['a1'] - plot_q['q1']
plot_q['a2q2'] = plot_q['a2'] - plot_q['q2']
plot_q['a3q3'] = plot_q['a3'] - plot_q['q3']
plot_q['ra1q1'] = plot_q['a1'] / plot_q['q1']
plot_q['ra2q2'] = plot_q['a2'] / plot_q['q2']
plot_q['ra3q3'] = plot_q['a3'] / plot_q['q3']

plot_q['c1s1'] = plot_q['c1'] - plot_q['s1']
plot_q['c2s2'] = plot_q['c2'] - plot_q['s2']
plot_q['c3s3'] = plot_q['c3'] - plot_q['s3']
plot_q['rc1s1'] = plot_q['c1'] / plot_q['s1']
plot_q['rc2s2'] = plot_q['c2'] / plot_q['s2']
plot_q['rc3s3'] = plot_q['c3'] / plot_q['s3']
plot_q['b1p11'] = plot_q['b1'] -  plot_q['p11']
plot_q['b2p21'] = plot_q['b2'] -  plot_q['p21']
plot_q['b3p31'] = plot_q['b3'] -  plot_q['p31']
plot_q['rb1p11'] = plot_q['b1'] /  plot_q['p11']
plot_q['rb2p21'] = plot_q['b2'] /  plot_q['p21']
plot_q['rb3p31'] = plot_q['b3'] /  plot_q['p31']

plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
fig1, ax1 = plt.subplots(1,2, figsize=(30,10))
#ax1[0].set_title('Market price', fontsize = 40, fontweight ="bold")
ax1[0].set_xlabel('price elasticity of demand'+' ('r'$1 /\beta$'+')', size=40)
ax1[0].set_ylabel('price', size=40)
ax1[0].tick_params(axis='both', which='major', labelsize=30)
sns.lineplot(data = plot_q, x='beta', y='P', label='market price', ax=ax1[0], linewidth=3, color='black')
ax1[0].legend(loc="upper left", prop={'size': 30})

#fig2, ax2 = plt.subplots(figsize=(10,5))
#ax1[1].set_title('Total production', fontsize = 40, fontweight ="bold")
ax1[1].set_xlabel('price elasticity of demand'+' ('r'$1 /\beta$'+')', size=40)
ax1[1].set_ylabel('quantity', size=40)
ax1[1].tick_params(axis='both', which='major', labelsize=30)
sns.lineplot(data = plot_q, x='beta', y='Q', label='total Q', ax=ax1[1], linewidth=3, color='black')
ax1[1].legend(loc="upper left", prop={'size': 30})

fig2, ax2 = plt.subplots(2,2, figsize=(30,22))
#ax2[0][0].set_title('Firm production', fontsize = 50, fontweight ="bold")
ax2[0][0].set_xlabel('price elasticity of demand'+' ('r'$1 /\beta$'+')', size=40)
ax2[0][0].set_ylabel('quantity', size=40)
ax2[0][0].tick_params(axis='both', which='major', labelsize=30)
sns.lineplot(data = plot_q, x='beta', y='q1', label=name_list[0], ax=ax2[0][0], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='q2', label=name_list[1], ax=ax2[0][0], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='q3', label=name_list[2], ax=ax2[0][0], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='q4', label=name_list[3], ax=ax2[0][0], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='zero', legend=False, ax=ax2[0][0], linewidth=3, linestyle='--', color='black')
#sns.lineplot(data = plot_q, x='beta', y='zero', label=name_list[3], ax=ax2[0], linewidth=3)
ax2[0][0].legend(loc="upper left", prop={'size': 30})

#fig4, ax4 = plt.subplots(figsize=(10,5))
#ax2[0][1].set_title('Firm profit', fontsize = 50, fontweight ="bold")
ax2[0][1].set_xlabel('price elasticity of demand'+' ('r'$1 /\beta$'+')', size=40)
ax2[0][1].set_ylabel('profit', size=40)
ax2[0][1].tick_params(axis='both', which='major', labelsize=30)
sns.lineplot(data = plot_q, x='beta', y='p1', label=name_list[0], ax=ax2[0][1], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='p2', label=name_list[1], ax=ax2[0][1], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='p3', label=name_list[2], ax=ax2[0][1], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='p4', label=name_list[3], ax=ax2[0][1], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='zero', legend=False, ax=ax2[0][1], linewidth=3, linestyle='--', color='black')
ax2[0][1].legend(loc="upper left", prop={'size': 30})

#ax2[1][0].set_title('Firm market share', fontsize = 50, fontweight ="bold")
ax2[1][0].set_xlabel('price elasticity of demand'+' ('r'$1 /\beta$'+')', size=40)
ax2[1][0].set_ylabel('market share', size=40)
ax2[1][0].tick_params(axis='both', which='major', labelsize=30)
sns.lineplot(data = plot_q, x='beta', y='s1', label=name_list[0], ax=ax2[1][0], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='s2', label=name_list[1], ax=ax2[1][0], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='s3', label=name_list[2], ax=ax2[1][0], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='s4', label=name_list[3], ax=ax2[1][0], linewidth=3)
#sns.lineplot(data = plot_q, x='beta', y='zero', legend=False, ax=ax2[1][0], linewidth=3, linestyle='--', color='black')
ax2[1][0].legend(loc="upper left", prop={'size': 30})

#ax2[1][1].set_title('Firm relative profit', fontsize = 50, fontweight ="bold")
ax2[1][1].set_xlabel('price elasticity of demand'+' ('r'$1 /\beta$'+')', size=40)
ax2[1][1].set_ylabel('relative profit', size=40)
ax2[1][1].tick_params(axis='both', which='major', labelsize=30)
sns.lineplot(data = plot_q, x='beta', y='p11', label=name_list[0], ax=ax2[1][1], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='p21', label=name_list[1], ax=ax2[1][1], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='p31', label=name_list[2], ax=ax2[1][1], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='p41', label=name_list[3], ax=ax2[1][1], linewidth=3)
#sns.lineplot(data = plot_q, x='beta', y='zero', legend=False, ax=ax2[1][1], linewidth=3, linestyle='--', color='black')
ax2[1][1].legend(loc="upper left", prop={'size': 30})

fig2.tight_layout()

fig3, ax3 = plt.subplots(2,2, figsize=(30,22))
#ax3[0][0].set_title('Firm production', fontsize = 50, fontweight ="bold")
ax3[0][0].set_xlabel('price elasticity of demand'+' ('r'$1 /\beta$'+')', size=40)
ax3[0][0].set_ylabel('quantity (n=3)', size=40)
ax3[0][0].tick_params(axis='both', which='major', labelsize=30)
sns.lineplot(data = plot_q, x='beta', y='a1', label=name_list[0], ax=ax3[0][0], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='a2', label=name_list[1], ax=ax3[0][0], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='a3', label=name_list[2], ax=ax3[0][0], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='zero', legend=False, ax=ax3[0][0], linewidth=3, linestyle='--', color='black')
#sns.lineplot(data = plot_q, x='beta', y='zero', label=name_list[3], ax=ax2[0], linewidth=3)
ax3[0][0].legend(loc="upper left", prop={'size': 30})

#fig4, ax4 = plt.subplots(figsize=(10,5))
#ax3[0][1].set_title('Firm production', fontsize = 50, fontweight ="bold")
ax3[0][1].set_xlabel('price elasticity of demand'+' ('r'$1 /\beta$'+')', size=40)
ax3[0][1].set_ylabel('quantity (n=4)', size=40)
ax3[0][1].tick_params(axis='both', which='major', labelsize=30)
sns.lineplot(data = plot_q, x='beta', y='q1', label=name_list[0], ax=ax3[0][1], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='q2', label=name_list[1], ax=ax3[0][1], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='q3', label=name_list[2], ax=ax3[0][1], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='zero', legend=False, ax=ax3[0][1], linewidth=3, linestyle='--', color='black')
ax3[0][1].legend(loc="upper left", prop={'size': 30})

#ax3[1][0].set_title('Firm production', fontsize = 50, fontweight ="bold")
ax3[1][0].set_xlabel('price elasticity of demand'+' ('r'$1 /\beta$'+')', size=40)
ax3[1][0].set_ylabel('quantity change in difference (post: n=3)', size=40)
ax3[1][0].tick_params(axis='both', which='major', labelsize=30)
sns.lineplot(data = plot_q, x='beta', y='a1q1', label=name_list[0], ax=ax3[1][0], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='a2q2', label=name_list[1], ax=ax3[1][0], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='a3q3', label=name_list[2], ax=ax3[1][0], linewidth=3)
#sns.lineplot(data = plot_q, x='beta', y='zero', legend=False, ax=ax2[1][0], linewidth=3, linestyle='--', color='black')
ax3[1][0].legend(loc="upper left", prop={'size': 30})

#ax3[1][1].set_title('Firm production', fontsize = 50, fontweight ="bold")
ax3[1][1].set_xlabel('price elasticity of demand'+' ('r'$1 /\beta$'+')', size=40)
ax3[1][1].set_ylabel('quantity change in ratio (post: n=3)', size=40)
ax3[1][1].tick_params(axis='both', which='major', labelsize=30)
sns.lineplot(data = plot_q, x='beta', y='ra1q1', label=name_list[0], ax=ax3[1][1], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='ra2q2', label=name_list[1], ax=ax3[1][1], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='ra3q3', label=name_list[2], ax=ax3[1][1], linewidth=3)
#sns.lineplot(data = plot_q, x='beta', y='zero', legend=False, ax=ax2[1][1], linewidth=3, linestyle='--', color='black')
ax3[1][1].legend(loc="upper left", prop={'size': 30})

fig3.tight_layout()

fig4, ax4 = plt.subplots(2,2, figsize=(30,22))
#ax4[0][0].set_title('Firm market share', fontsize = 50, fontweight ="bold")
ax4[0][0].set_xlabel('price elasticity of demand'+' ('r'$1 /\beta$'+')', size=40)
ax4[0][0].set_ylabel('market share (n=3)', size=40)
ax4[0][0].tick_params(axis='both', which='major', labelsize=30)
sns.lineplot(data = plot_q, x='beta', y='c1', label=name_list[0], ax=ax4[0][0], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='c2', label=name_list[1], ax=ax4[0][0], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='c3', label=name_list[2], ax=ax4[0][0], linewidth=3)
#sns.lineplot(data = plot_q, x='beta', y='zero', legend=False, ax=ax3[0][0], linewidth=3, linestyle='--', color='black')
#sns.lineplot(data = plot_q, x='beta', y='zero', label=name_list[3], ax=ax2[0], linewidth=3)
ax4[0][0].legend(loc="upper left", prop={'size': 30})

#fig4, ax4 = plt.subplots(figsize=(10,5))
#ax4[0][1].set_title('Firm production', fontsize = 50, fontweight ="bold")
ax4[0][1].set_xlabel('price elasticity of demand'+' ('r'$1 /\beta$'+')', size=40)
ax4[0][1].set_ylabel('market share (n=4)', size=40)
ax4[0][1].tick_params(axis='both', which='major', labelsize=30)
sns.lineplot(data = plot_q, x='beta', y='s1', label=name_list[0], ax=ax4[0][1], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='s2', label=name_list[1], ax=ax4[0][1], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='s3', label=name_list[2], ax=ax4[0][1], linewidth=3)
#sns.lineplot(data = plot_q, x='beta', y='zero', legend=False, ax=ax3[0][1], linewidth=3, linestyle='--', color='black')
ax4[0][1].legend(loc="upper left", prop={'size': 30})

#ax4[1][0].set_title('Firm production', fontsize = 50, fontweight ="bold")
ax4[1][0].set_xlabel('price elasticity of demand'+' ('r'$1 /\beta$'+')', size=40)
ax4[1][0].set_ylabel('market share change in difference (post: n=3)', size=40)
ax4[1][0].tick_params(axis='both', which='major', labelsize=30)
sns.lineplot(data = plot_q, x='beta', y='c1s1', label=name_list[0], ax=ax4[1][0], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='c2s2', label=name_list[1], ax=ax4[1][0], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='c3s3', label=name_list[2], ax=ax4[1][0], linewidth=3)
#sns.lineplot(data = plot_q, x='beta', y='zero', legend=False, ax=ax2[1][0], linewidth=3, linestyle='--', color='black')
ax4[1][0].legend(loc="upper left", prop={'size': 30})

#ax4[1][1].set_title('Firm production', fontsize = 50, fontweight ="bold")
ax4[1][1].set_xlabel('price elasticity of demand'+' ('r'$1 /\beta$'+')', size=40)
ax4[1][1].set_ylabel('market share change in ratio (post: n=3)', size=40)
ax4[1][1].tick_params(axis='both', which='major', labelsize=30)
sns.lineplot(data = plot_q, x='beta', y='rc1s1', label=name_list[0], ax=ax4[1][1], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='rc2s2', label=name_list[1], ax=ax4[1][1], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='rc3s3', label=name_list[2], ax=ax4[1][1], linewidth=3)
#sns.lineplot(data = plot_q, x='beta', y='zero', legend=False, ax=ax2[1][1], linewidth=3, linestyle='--', color='black')
ax4[1][1].legend(loc="upper left", prop={'size': 30})

fig4.tight_layout()

fig5, ax5 = plt.subplots(2,2, figsize=(30,22))
#ax5[0][0].set_title('Firm market share', fontsize = 50, fontweight ="bold")
ax5[0][0].set_xlabel('price elasticity of demand'+' ('r'$1 /\beta$'+')', size=40)
ax5[0][0].set_ylabel('profit (n=3)', size=40)
ax5[0][0].tick_params(axis='both', which='major', labelsize=30)
sns.lineplot(data = plot_q, x='beta', y='b1', label=name_list[0], ax=ax5[0][0], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='b2', label=name_list[1], ax=ax5[0][0], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='b3', label=name_list[2], ax=ax5[0][0], linewidth=3)
#sns.lineplot(data = plot_q, x='beta', y='zero', legend=False, ax=ax3[0][0], linewidth=3, linestyle='--', color='black')
#sns.lineplot(data = plot_q, x='beta', y='zero', label=name_list[3], ax=ax2[0], linewidth=3)
ax5[0][0].legend(loc="upper left", prop={'size': 30})

#fig4, ax4 = plt.subplots(figsize=(10,5))
#ax5[0][1].set_title('Firm production', fontsize = 50, fontweight ="bold")
ax5[0][1].set_xlabel('price elasticity of demand'+' ('r'$1 /\beta$'+')', size=40)
ax5[0][1].set_ylabel('profit (n=4)', size=40)
ax5[0][1].tick_params(axis='both', which='major', labelsize=30)
sns.lineplot(data = plot_q, x='beta', y='p11', label=name_list[0], ax=ax5[0][1], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='p21', label=name_list[1], ax=ax5[0][1], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='p31', label=name_list[2], ax=ax5[0][1], linewidth=3)
#sns.lineplot(data = plot_q, x='beta', y='zero', legend=False, ax=ax3[0][1], linewidth=3, linestyle='--', color='black')
ax5[0][1].legend(loc="upper left", prop={'size': 30})

#ax5[1][0].set_title('Firm production', fontsize = 50, fontweight ="bold")
ax5[1][0].set_xlabel('price elasticity of demand'+' ('r'$1 /\beta$'+')', size=40)
ax5[1][0].set_ylabel('profit change in difference (post: n=3)', size=40)
ax5[1][0].tick_params(axis='both', which='major', labelsize=30)
sns.lineplot(data = plot_q, x='beta', y='b1p11', label=name_list[0], ax=ax5[1][0], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='b2p21', label=name_list[1], ax=ax5[1][0], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='b3p31', label=name_list[2], ax=ax5[1][0], linewidth=3)
#sns.lineplot(data = plot_q, x='beta', y='zero', legend=False, ax=ax2[1][0], linewidth=3, linestyle='--', color='black')
ax5[1][0].legend(loc="upper left", prop={'size': 30})

#ax5[1][1].set_title('Firm production', fontsize = 50, fontweight ="bold")
ax5[1][1].set_xlabel('price elasticity of demand'+' ('r'$1 /\beta$'+')', size=40)
ax5[1][1].set_ylabel('profit change in ratio (post: n=3)', size=40)
ax5[1][1].tick_params(axis='both', which='major', labelsize=30)
sns.lineplot(data = plot_q, x='beta', y='rb1p11', label=name_list[0], ax=ax5[1][1], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='rb2p21', label=name_list[1], ax=ax5[1][1], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='rb3p31', label=name_list[2], ax=ax5[1][1], linewidth=3)
#sns.lineplot(data = plot_q, x='beta', y='zero', legend=False, ax=ax2[1][1], linewidth=3, linestyle='--', color='black')
ax5[1][1].legend(loc="upper left", prop={'size': 30})

fig5.tight_layout()

if c_list[3] >= (6.*(c_list[0]+c_list[1]+c_list[2]))/(5.*3.):
    st.write("Your input value violates condition (1)")
elif c_list[2] >= (4.*(c_list[0]+c_list[1]))/(3.*2.):
    st.write("Your input value violates condition (2)")

st.subheader('Panel A: Market outcomes')
st.pyplot(fig1)
    
st.subheader('Panel B: Firm-level outcomes')
st.pyplot(fig2)
    
st.subheader('Panel C: Before and after comparison (n = 4 -> n = 3)')
st.pyplot(fig3)
st.pyplot(fig4)
st.pyplot(fig5)


#if (c_list[3] < c_bar4*8./7.):
    #results


#else:
#    st.write(c_bar4*8./7.)
#    st.subheader('Panel A: Market outcomes!')
    
#st.pyplot(fig3)

#st.subheader('Firm profit')
#st.pyplot(fig4)

#st.subheader('Raw data2')
#st.line_chart(chart_data)
#st.pyplot(fig)


