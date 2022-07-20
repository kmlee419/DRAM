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

c_list[0] = st.sidebar.number_input('삼성전자', value=1.0, step=0.01)
name_list[0] = "삼성전자"
#st.write('The current number is ', c_list[0])

c_list[1] = st.sidebar.number_input('Insert a c2 number', min_value = c_list[0], step=0.01)
name_list[1] = "c2"
c_list[2] = st.sidebar.number_input('Insert a c3 number', min_value = c_list[1], step=0.01)
name_list[2] = "c3"
c_list[3] = st.sidebar.number_input('Insert a c4 number', min_value = c_list[2], step=0.01)
name_list[3] = "c4"

c_bar = sum(c_list)/len(c_list)

mu = st.sidebar.slider('Select a mu value',
                           0.0, 20.0, 8.624)

D = 1

beta_range = np.arange(0.25, 4, 0.01)

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
fig1, ax1 = plt.subplots(figsize=(10,5))
ax1.set_xlabel(r'$1 /\beta$', size=20)
ax1.set_ylabel('q', size=20)
sns.lineplot(data = plot_q, x='beta', y=0, label=name_list[0], ax=ax1, linewidth=3)
sns.lineplot(data = plot_q, x='beta', y=1, label=name_list[1], ax=ax1, linewidth=3)
sns.lineplot(data = plot_q, x='beta', y=2, label=name_list[2], ax=ax1, linewidth=3)
ax1.legend(loc="upper left", prop={'size': 15})

fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.set_xlabel(r'$1 /\beta$', size=20)
ax2.set_ylabel('Q', size=20)
sns.lineplot(data = plot_q, x='beta', y='Q', label='total Q', ax=ax2, linewidth=3)
ax2.legend(loc="upper left", prop={'size': 15})

fig3, ax3 = plt.subplots(figsize=(10,5))
ax3.set_xlabel(r'$1 /\beta$', size=20)
ax3.set_ylabel('P', size=20)
sns.lineplot(data = plot_q, x='beta', y='P', label='Price P', ax=ax3, linewidth=3)
ax3.legend(loc="upper left", prop={'size': 15})

fig4, ax4 = plt.subplots(figsize=(10,5))
ax4.set_xlabel(r'$1 /\beta$', size=20)
ax4.set_ylabel('profit', size=20)
sns.lineplot(data = plot_q, x='beta', y='p1', label=name_list[0], ax=ax4, linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='p2', label=name_list[1], ax=ax4, linewidth=3)
sns.lineplot(data = plot_q, x='beta', y='p3', label=name_list[2], ax=ax4, linewidth=3)
ax4.legend(loc="upper left", prop={'size': 15})

st.subheader('q result')
st.pyplot(fig1)

st.subheader('Q result')
st.pyplot(fig2)

st.subheader('P result')
st.pyplot(fig3)

st.subheader('profit result')
st.pyplot(fig4)

#st.subheader('Raw data2')
#st.line_chart(chart_data)
#st.pyplot(fig)
