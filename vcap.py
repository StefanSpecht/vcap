# coding: utf-8
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib.utils import ImageReader
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from datetime import date, datetime, timedelta
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, inch
import argparse
#import pickle


def predict_capacity (datafile, predict_years_max, args):

    repo_data_all = []      #list of repo_data dictionaries

    # Load dataframe from csv report
    df = pd.read_csv(datafile, header=0, index_col=0, parse_dates=True, dayfirst=True)
    df['UsedGB'] = df['StorageTotal'] - df['StorageFree']
    df = df[['Name', 'UsedGB', 'StorageTotal']]

    # Get list of repository names
    names = df['Name']
    names = names.unique()

    #forecast timeseries for each repository
    for name in names:

        ## TEST
        #name = "ES1_EE2203_Backup_01"
        ## /TEST
        if not args.quiet:
            print('Processing {}'.format(name))
        df_current = df[df['Name'] == name]
        series = pd.Series(df_current['UsedGB'])

        #Re-index series and forward-fill missing values
        date_range = pd.date_range(series.index[1].date(), series.index[-1].date())
        series = series.reindex(date_range)
        series.fillna(method='ffill', inplace=True)

        soft_threshold_reached = False
        repo_data = {
            'name': name,
            'max_capacity': df_current['StorageTotal'].iloc[-1],
            'used_gb': df_current['UsedGB'].iloc[-1]
        }

        try:
            # fit ARIMA model and print summary
            model = ARIMA(series, order=(6,1,1))
            model_fit = model.fit(disp=0)
            if args.verbose:
                print(model_fit.summary())
            history = [x for x in series.values]
            hard_threshold = df_current['StorageTotal'].iloc[-1]
            soft_threshold = hard_threshold * 0.85
            current_date = series.tail(1).index.date
            predicted_series = pd.Series()
            predicted_series = predicted_series.append(series[-1:])
            enddate = current_date + timedelta(days=(predict_years_max * 365))

            while True:
                try:
                    current_date += timedelta(days=1)
                    model = ARIMA(history, order=(6,1,1))
                    model_fit = model.fit(disp=0)
                    output = model_fit.forecast(steps=1)
                    yhat = output[0]
                    if args.verbose:
                        print('predicted=%f, date=%s' % (yhat, current_date))
                    predicted_series = predicted_series.append(pd.Series(yhat, index=current_date))
                    history.append(yhat[0])
                except:
                    if args.verbose:
                        print("Error predicting next value")

                if (yhat >= soft_threshold and soft_threshold_reached == False):
                    soft_threshold_reached = True
                    repo_data['soft_threshold_date'] = current_date[0]
                if (yhat >= hard_threshold):
                    repo_data['hard_threshold_date'] = current_date[0]
                if (yhat >= hard_threshold) or (current_date > enddate) or (yhat == 0):
                    repo_data['did_fit'] = True
                    break

            #scale GB to TB
            if hard_threshold > 1024:
                series = series/1024
                predicted_series = predicted_series/1024
                hard_threshold = hard_threshold/1024
                soft_threshold = soft_threshold/1024
                ylabel = 'Used [TB]'
            else:
                ylabel = 'Used [GB]'

            fig = plt.figure(figsize=(17*cm/inch, 17*cm/inch))
            plt.title(name)
            plt.plot(series)
            plt.plot(predicted_series, color='red')
            plt.axhline(y=hard_threshold, color='r')
            plt.axhline(y=soft_threshold, color='y', ls=':')
            plt.ylabel(ylabel)
        except:
            if hard_threshold > 1024:
                series = series/1024
                predicted_series = predicted_series/1024
                hard_threshold = hard_threshold/1024
                soft_threshold = soft_threshold/1024
                ylabel = 'Used [TB]'
            else:
                ylabel = 'Used [GB]'
            repo_data['did_fit'] = False
            fig = plt.figure(figsize=(17*cm/inch, 17*cm/inch))
            plt.title(name + '(model did not fit)')
            plt.plot(series)
            plt.axhline(y=hard_threshold, color='r')
            plt.axhline(y=soft_threshold, color='y', ls=':')
            plt.ylabel(ylabel)

        imgdata = BytesIO()
        fig.savefig(imgdata, format='png')
        imgdata.seek(0) #rewind data
        repo_data['figure'] = imgdata

        repo_data_all.append(repo_data)

        ## TEST
        break
        ## /TEST

    return repo_data_all

def generate_report(repo_data_all, reportfile, logofile):
    repo_data_all.sort(key = lambda repo_data: repo_data.get('hard_threshold_date', date.today() + timedelta(days=20*365)))

    width, height = A4
    margin_left = 1.5*cm
    margin_right = 1.5*cm
    margin_top = 2*cm
    margin_bottom = 2*cm
    tab = 1*cm
    name_tab = 6*tab
    capacity_tab = 2*tab
    used_tab = 2*tab
    percent_tab = 2*tab
    soft_threshold_tab = 3*tab

    pdf = canvas.Canvas(reportfile, pagesize = A4)

    #Generate title & logo
    pdf.drawImage(logofile, width - (3.5 * cm), height - (4.7 * cm), width = 3*cm, preserveAspectRatio = True, mask='auto')
    pdf.setFont('Helvetica-Bold', 28 )
    pdf.setFillColor("green")
    pdf.setStrokeColor("green")
    y = height - (2.6 * cm)
    pdf.drawCentredString(width/2, y, 'Veeam Capacity Report')
    y = y - 0.4*cm
    pdf.setLineWidth(2.0)
    pdf.line(width/4 - 0.4*cm, y, width/4*3 + 0.4*cm, y)

    #Generate summary table
    #title
    y = y - 3*cm
    pdf.setFont('Helvetica-Bold', 12)
    pdf.setFillColor("black")
    pdf.setStrokeColor("black")
    pdf.drawString(margin_left, y, "SUMMARY")
    y = y - 0.1*cm
    pdf.setLineWidth(1.0)
    pdf.line(margin_left, y, margin_left + 2.2*cm, y)
    y = y - 1.5*cm

    #header
    pdf.setLineWidth(0.5)
    pdf.setFont('Helvetica-Bold', 11)
    x = margin_left
    pdf.drawString(x, y, 'Name')
    x = x + name_tab
    pdf.drawString(x, y, 'Capacity')
    x = x + capacity_tab
    pdf.drawString(x, y, 'Used')
    x = x + used_tab
    pdf.drawString(x, y, 'Used [%]')
    x = x + percent_tab
    pdf.drawString(x, y, '85% reached')
    x = x + soft_threshold_tab
    pdf.drawString(x, y, '100% reached')
    pdf.setFont('Helvetica', 11)

    y = y - 0.2*cm
    pdf.line(margin_left, y, width - margin_right, y)
    y = y - 0.05*cm
    pdf.line(margin_left, y, width - margin_right, y)
    y = y - 0.4*cm


    #Create entries
    for repo_data in repo_data_all:

        if ('soft_threshold_date' in repo_data) and ((repo_data['soft_threshold_date'] - date.today()).days <= 90):
            pdf.setFillColor("orange")
            pdf.setStrokeColor("orange")

        if ('hard_threshold_date' in repo_data) and ((repo_data['hard_threshold_date'] - date.today()).days <= 90):
            pdf.setFillColor("red")
            pdf.setStrokeColor("red")

        x = margin_left
        pdf.drawString(x, y, repo_data['name'])
        x = x + name_tab
        if (repo_data['max_capacity'] < 1024):
            pdf.drawString(x, y, '{:,.0f} GB'.format(repo_data['max_capacity']) )
            x = x + capacity_tab
            pdf.drawString(x, y, '{:,.0f} GB'.format(repo_data['used_gb']))
        else:
            pdf.drawString(x, y, '{:,.2f} TB'.format(repo_data['max_capacity'] / 1024) )
            x = x + capacity_tab
            pdf.drawString(x, y, '{:,.2f} TB'.format(repo_data['used_gb'] / 1024))
        x = x + used_tab
        pdf.drawString(x, y, '{0:.0%}'.format(repo_data['used_gb'] / repo_data['max_capacity']))
        x = x + percent_tab
        if 'soft_threshold_date' in repo_data:
            pdf.drawString(x, y, '{:%d.%m.%y}'.format(repo_data['soft_threshold_date'] ))
        x = x + soft_threshold_tab
        if 'hard_threshold_date' in repo_data:
            pdf.drawString(x, y, '{:%d.%m.%y}'.format(repo_data['hard_threshold_date'] ))

        pdf.setFillColor("black")
        pdf.setStrokeColor("black")

        y = y - 0.2*cm
        pdf.line(margin_left, y, width - margin_right, y)
        y = y - 0.4*cm

        #if page is full, create a new page and re-draw header
        if y <= margin_bottom:
            pdf.showPage()
            y = height - margin_top
            pdf.setLineWidth(0.5)
            pdf.setFont('Helvetica-Bold', 11)
            x = margin_left
            pdf.drawString(x, y, 'Name')
            x = x + name_tab
            pdf.drawString(x, y, 'Capacity')
            x = x + capacity_tab
            pdf.drawString(x, y, 'Used')
            x = x + used_tab
            pdf.drawString(x, y, 'Used [%]')
            x = x + percent_tab
            pdf.drawString(x, y, '85% reached')
            x = x + soft_threshold_tab
            pdf.drawString(x, y, '100% reached')
            pdf.setFont('Helvetica', 11)

            y = y - 0.2*cm
            pdf.line(margin_left, y, width - margin_right, y)
            y = y - 0.05*cm
            pdf.line(margin_left, y, width - margin_right, y)
            y = y - 0.4*cm


    #Create detail pages
    figsize_x = 10*cm
    figsize_y = 10*cm
    figspacer = 2*cm
    y = height - margin_top
    x = margin_left
    first_page = True

    for repo_data in repo_data_all:
        if y <= margin_bottom or first_page == True:
            pdf.showPage()
            y = height - margin_top
            pdf.setFont('Helvetica-Bold', 12)
            pdf.drawString(x, y, "CAPACITY FORECAST BY REPOSITORY")
            y = y - 0.1*cm
            pdf.setLineWidth(1.0)
            pdf.line(x, y, x + 8.3*cm, y)
            y = y - 1.5*cm
            y = y - figsize_y
            first_page = False

        pdf.setFont('Helvetica', 11)
        Image = ImageReader(repo_data['figure'])
        pdf.drawImage(Image, x, y, figsize_x, figsize_y)
        textspacer = 1.3*cm
        x1 = x + figsize_x + 1*cm
        x2 = x1 + 2*cm
        y1 = y + figsize_y - 1.9*textspacer
        pdf.drawString(x1, y1, 'Name:')
        pdf.drawString(x2,y1, '{}'.format(repo_data['name']))
        y1 = y1 - textspacer
        if (repo_data['max_capacity'] <= 1024):
            pdf.drawString(x1, y1, 'Capacity:')
            pdf.drawString(x2, y1, '{:,.0f} GB'.format(repo_data['max_capacity']) )
            y1 = y1 - textspacer
            pdf.drawString(x1, y1, 'Used:')
            pdf.drawString(x2, y1, '{:,.0f} GB ({:.0%})'.format(repo_data['used_gb'], repo_data['used_gb'] / repo_data['max_capacity']))
        else:
            pdf.drawString(x1, y1, 'Capacity:')
            pdf.drawString(x2, y1, '{:,.2f} TB'.format(repo_data['max_capacity']/1024 ))
            y1 = y1 - textspacer
            pdf.drawString(x1, y1, 'Used:')
            pdf.drawString(x2, y1, '{:,.2f} TB ({:.0%})'.format(repo_data['used_gb']/1024, repo_data['used_gb'] / repo_data['max_capacity']))
        y1 = y1 - textspacer
        if 'soft_threshold_date' in repo_data:
            pdf.drawString(x1, y1, '85% on:')
            pdf.drawString(x2, y1, '{:%d.%m.%y}'.format(repo_data['soft_threshold_date'] ))
        else:
            pdf.drawString(x1, y1, '85% on:')
            pdf.drawString(x2, y1, 'N/A')
        y1 = y1 - textspacer
        if 'hard_threshold_date' in repo_data:
            pdf.drawString(x1, y1, '100% on:')
            pdf.drawString(x2, y1, '{:%d.%m.%y}'.format(repo_data['hard_threshold_date'] ))
        else:
            pdf.drawString(x1, y1, '100% on:')
            pdf.drawString(x2, y1, 'N/A')


        y = y - figspacer - figsize_y

    pdf.save()

def main ():
    #parse arguments
    parser = argparse.ArgumentParser(description="Generate a Veeam Capacity Report")
    parsegroup = parser.add_mutually_exclusive_group()
    parsegroup.add_argument("-v", "--verbose", action="store_true", help="show verbose text output")
    parsegroup.add_argument("-q", "--quiet", action="store_true", help="supress any text output")
    parser.add_argument("inputfile", help="input file (csv)")
    parser.add_argument("-o", "--output", help="output file (pdf)", default="VeeamCapacityReport_{:%Y%m%d}.pdf".format(date.today()))
    parser.add_argument("-y", "--years", type=int, default=2, help="forecast time in years")
    args = parser.parse_args()

    #Define parameters
    datafile = args.inputfile
    reportfile = args.output
    logofile = './img/logo.png'
    predict_years_max = args.years  #if forecast does not reach 100% capacity after x
                                    #years, prediction will abort for given repository

    repo_data_all = predict_capacity(datafile, predict_years_max, args)

    #TEST - persist data
    #persistent_data = open('repo_data_all.pkl', 'wb')
    #pickle.dump(repo_data_all, persistent_data)
    #persistent_data.close()

    # TEST - load data
    #persistent_data = open('repo_data_all.pkl', 'rb')
    #repo_data_all = pickle.load(persistent_data)
    #persistent_data.close()
    # /TEST

    generate_report(repo_data_all, reportfile, logofile)

if __name__ == '__main__':
    main()
