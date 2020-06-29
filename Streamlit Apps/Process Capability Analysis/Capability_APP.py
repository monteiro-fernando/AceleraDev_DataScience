import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

def kolmogorov_smirnov_critico(n):
    # table of critical values for the kolmogorov-smirnov test - 95% confidence
    # Source: https://www.soest.hawaii.edu/GG/FACULTY/ITO/GG413/K_S_Table_one_Sample.pdf
    # Source: http://www.real-statistics.com/statistics-tables/kolmogorov-smirnov-table/
    # alpha = 0.05 (95% confidential level)
    
    if n <= 40:
        # valores entre 1 e 40
        kolmogorov_critico = [0.97500, 0.84189, 0.70760, 0.62394, 0.56328, 0.51926, 0.48342, 0.45427, 0.43001, 0.40925, 
                      0.39122, 0.37543, 0.36143, 0.34890, 0.33760, 0.32733, 0.31796, 0.30936, 0.30143, 0.29408, 
                      0.28724, 0.28087, 0.27490, 0.26931, 0.26404, 0.25907, 0.25438, 0.24993, 0.24571, 0.24170, 
                      0.23788, 0.23424, 0.23076, 0.22743, 0.22425, 0.22119, 0.21826, 0.21544, 0.21273, 0.21012]
        ks_critico = kolmogorov_critico[n - 1]
    elif n > 40:
        # valores acima de 40:
        kolmogorov_critico = 1.36/(np.sqrt(n))
        ks_critico = kolmogorov_critico
    else:
        pass            
            
    return ks_critico

def Cp(mylist, ls, li):
    arr = np.array(mylist)
    arr = arr.ravel()
    sigma = np.std(arr)
    Cp = float(ls - li) / (6*sigma)
    return Cp


def Cpk(mylist, ls, li):
    arr = np.array(mylist)
    arr = arr.ravel()
    sigma = np.std(arr)
    m = np.mean(arr)

    Cps = float(ls - m) / (3*sigma)
    Cpi = float(m - li) / (3*sigma)
    Cpk = np.min([Cps, Cpi])
    return Cpk

def main():
    st.image(['logo.png','fatec.png'], width = 330)
    st.title('Process Capability Analysis')
    st.text('By Fernando Monteiro')

    options = ['View Raw Data', 'Remove Outliers', 'Normality Test', 'Process Capability']
    choice = st.sidebar.selectbox('Select Activities', options)
    st.markdown('Process capability compares the output of an in-control process to the specification limits by using capability indices. The comparison is made by forming the ratio of the spread between the process specifications (the specification "width") to the spread of the process values, as measured by 6 process standard deviation units (the process "width").')
    st.markdown('**If you do not have your own data, click on the link bellow and download our sample data to test.**')
    '''
    [Download](https://drive.google.com/file/d/15jYCdZN4m_2b_UnIA4a7h2QnvNvvlQpV/view?usp=sharing)
    '''
    file  = st.file_uploader('Load your sample here: (.csv)', type = 'csv',)
    if file is not None:
        df = pd.read_csv(file, delimiter=";", decimal=",")
        df = pd.DataFrame(data = df)

        if choice == 'View Raw Data':
            st.markdown('**Data Preview:**')
            number = st.slider('Choose the number of rows you want to see:', min_value=1, max_value=50, key = 1)
            st.dataframe(df.head(number))

            if st.checkbox('Show Sample Size'):
                st.markdown('**Sample Size:**')
                st.write(df.shape[0])
            
            if st.checkbox('Show Mean'):
                mean = float(round(df.mean(), 4))
                st.markdown('**Mean:**')
                st.write(mean)

            if st.checkbox('Show Standard Deviation'):
                std = float(round(df.std(), 4))
                st.markdown('**Standard Deviation:**')
                st.write(std)

            if st.checkbox('Show Minimum Value'):
                min = float(round(df.min(),4))
                st.markdown('**Minimum Value:**')
                st.write(min)

            if st.checkbox('Show Maximum Value'):
                max = float(round(df.max(),4))
                st.markdown('**Maximum Value:**')
                st.write(max)

            if st.checkbox ('Show Histogram'):
                st.markdown('**Histogram**')
                number = st.slider('Choose bin size:', min_value=5, max_value=20, key = 2)
                fig = px.histogram(df, x = df.columns[0], nbins = number)
                st.plotly_chart(fig, use_container_width=True)
        
        if choice == 'Remove Outliers':
            st.subheader('After Outliers Removal')
            df_out = remove_outlier(df, df.columns[0])
            st.markdown('**Data Preview**')
            number = st.slider('Choose the number of rows you want to see:', min_value=1, max_value=50, key =3)
            st.dataframe(df_out.head(number))

            if st.checkbox('Show Sample Size'):
                st.markdown('**Sample Size:**')
                st.write(df_out.shape[0])
            
            if st.checkbox('Show Mean'):
                mean = float(round(df_out.mean(), 4))
                st.markdown('**Mean:**')
                st.write(mean)

            if st.checkbox('Show Standard Deviation'):
                std = float(round(df_out.std(), 4))
                st.markdown('**Standard Deviation:**')
                st.write(std)

            if st.checkbox('Show Minimum Value'):
                min = float(round(df_out.min(),4))
                st.markdown('**Minimum Value:**')
                st.write(min)

            if st.checkbox('Show Maximum Value'):
                max = float(round(df_out.max(),4))
                st.markdown('**Maximum Value:**')
                st.write(max)

            if st.checkbox ('Show Histogram'):
                st.markdown('**Histogram**')
                number = st.slider('Choose bin size:', min_value=5, max_value=20, key = 2)
                fig = px.histogram(df_out, x = df_out.columns[0], nbins = number, )
                st.plotly_chart(fig, use_container_width=True)

        if choice == 'Normality Test':
            df_out = remove_outlier(df, df.columns[0])
            mean = float(round(np.mean(df_out), 4))
            std = float(round(np.std(df_out, ddof = 1), 4))
            st.subheader('Kolmogorov-Smirnov Normality Test')

            if st.checkbox('Show Mean'):
                st.markdown('**Mean:**')
                st.write(mean)

            if st.checkbox('Show Standard Deviation'):
                st.markdown('**Standard Deviation:**')
                st.write(std)

            if st.checkbox('Apply Normality Test'):
                # Checking the critical value of the Kolmogorov-Smirnov test
                ks_critico = kolmogorov_smirnov_critico(df_out.shape[0])
                # Calculating the value of the Kolmogorov-Smirnov statistic for the data
                ks_stat, ks_p_valor = stats.kstest(df_out, cdf='norm', args=(mean, std), N = df_out.shape[0])
                st.markdown("At 95% confidence level, the critical value of the Kolmogorov-Smirnov test is = " + str(ks_critico))
                st.markdown("The calculated value of the Kolmogorov-Smirnov test is = " + str(ks_stat))
                # Conclusion
                if ks_critico >= ks_stat:
                    st.markdown("**At 95% confidence level, we have NO evidence to reject the hypothesis of data normality, according to the Kolmogorov-Smirnov test.**")
                else:
                    st.markdown("**At 95% confidence level, we have evidence to reject the hypothesis of data normality, according to the Kolmogorov-Smirnov test.**")
            
        if choice == 'Process Capability':
            df_out = remove_outlier(df, df.columns[0])    
            q1 = df_out[df_out.columns[0]].quantile(0.25)
            q3 = df_out[df_out.columns[0]].quantile(0.75)
            iqr = q3-q1 #Interquartile range
            li = q1-1.5*iqr
            ls = q3+1.5*iqr
            mylist = df_out
            mylist = mylist.values.tolist()
            cp = Cp(mylist, ls, li)
            cpk = Cpk(mylist, ls, li)

            if st.checkbox('Calculate Cp and Cpk'):
                st.markdown('**Cp:**')
                st.write(cp)
                st.markdown('**Cpk:**')
                st.write(cpk)


            
        



if __name__ == '__main__':
	main()