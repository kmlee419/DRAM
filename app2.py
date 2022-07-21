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

c_bar = sum(c_list)/len(c_list)

mu = st.sidebar.slider('Insert the market size parameter (mu)',
                           1.0, 20.0, 8.624)

D = 1

if (c_list[3]-c_bar)==0:
    beta_max = 1
else:
    beta_max = (c_list[3]/(c_list[3]-c_bar))*(1/n)

beta_range = np.arange(0.25, beta_max, 0.01)

#results
plot_q = pd.DataFrame(beta_range, columns=['beta'])

for i in range(0, n):
    
    values = q_star(c_list[i], c_bar, mu, D, beta_range, n)
    plot_q[i] = values

plot_q['Q'] = plot_q.iloc[:, 1:].sum(axis=1)
plot_q['P'] = c_bar*n/(n-plot_q['beta'])
plot_q['p1'] = (plot_q['P']-c_list[0])*plot_q[0]
plot_q['p2'] = (plot_q['P']-c_list[1])*plot_q[1]
plot_q['p3'] = (plot_q['P']-c_list[2])*plot_q[2]
plot_q['p4'] = (plot_q['P']-c_list[3])*plot_q[3]

plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
fig1, ax1 = plt.subplots(1,2, figsize=(30,10))
ax1[0].set_title('Production', fontsize = 50, fontweight ="bold")
ax1[0].set_xlabel('price elasticity of demand'+' ('r'$1 /\beta$'+')', size=40)
ax1[0].set_ylabel('quantity', size=40)
ax1[0].tick_params(axis='both', which='major', labelsize=30)
sns.lineplot(data = plot_q, x='beta', y=0, label=name_list[0], ax=ax1[0], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y=1, label=name_list[1], ax=ax1[0], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y=2, label=name_list[2], ax=ax1[0], linewidth=3)
ax1[0].legend(loc="upper left", prop={'size': 30})

#fig2, ax2 = plt.subplots(figsize=(10,5))
ax1[1].set_title('Total production', fontsize = 50, fontweight ="bold")
ax1[1].set_xlabel('price elasticity of demand'+' ('r'$1 /\beta$'+')', size=40)
ax1[1].set_ylabel('quantity', size=40)
ax1[1].tick_params(axis='both', which='major', labelsize=30)
sns.lineplot(data = plot_q, x='beta', y='Q', label='total Q', ax=ax1[1], linewidth=3)
ax1[1].legend(loc="upper left", prop={'size': 30})

fig2, ax2 = plt.subplots(1,2, figsize=(30,10))
ax2[0].set_title('Market price', fontsize = 50, fontweight ="bold")
ax2[0].set_xlabel('price elasticity of demand'+' ('r'$1 /\beta$'+')', size=40)
ax2[0].set_ylabel('price', size=40)
ax2[0].tick_params(axis='both', which='major', labelsize=30)
sns.lineplot(data = plot_q, x='beta', y='P', label='market price', ax=ax2[0], linewidth=3)
ax2[0].legend(loc="upper left", prop={'size': 30})

#fig4, ax4 = plt.subplots(figsize=(10,5))
ax2[1].set_title('Firm profit', fontsize = 50, fontweight ="bold")
ax2[1].set_xlabel('price elasticity of demand'+' ('r'$1 /\beta$'+')', size=40)
ax2[1].set_ylabel('profit', size=40)
ax2[1].tick_params(axis='both', which='major', labelsize=30)
sns.lineplot(data = plot_q, x='beta', y='p1', label=name_list[0], ax=ax2[1], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='p2', label=name_list[1], ax=ax2[1], linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='p3', label=name_list[2], ax=ax2[1], linewidth=3)
ax2[1].legend(loc="upper left", prop={'size': 30})

#st.subheader('Production')
st.subheader('    ')
st.pyplot(fig1)

#st.subheader('Total production')
st.subheader('    ')
st.pyplot(fig2)

#st.subheader('Market price')
#st.pyplot(fig3)

#st.subheader('Firm profit')
#st.pyplot(fig4)

#st.subheader('Raw data2')
#st.line_chart(chart_data)
#st.pyplot(fig)
