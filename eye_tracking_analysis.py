import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import PIL
from PIL import Image


class EyeTrackerAnalyser:
    def __init__(self, path_to_data: str):

        self._path_to_data = path_to_data

    def analyse_data(self, set_mode: str):
        meta_data = pd.read_csv(os.path.join(self._path_to_data, 'metadata.csv'))
        analys_data = meta_data.loc[meta_data['dataset'] == 'train']
        fig = px.pie(analys_data, values='recording_id', names='device_name', title='Recordings per device - train')
        fig.update_traces(textposition='outside', textinfo='percent+label', textfont_size=20)
        fig.update_layout(legend=dict(
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),font=dict(size=20))
        fig.show()

    def calc_x_y_prediction_table(self, set_mode: str):
        pred_data = pd.read_csv(os.path.join(self._path_to_data, 'predicted_data.csv'))
        screen_size_panda = pd.read_csv(r'/mnt/2ef93ccf-c66e-4beb-95ba-24011e8fee18/TAMAR/screen_size.csv')
        # pred_data_panda = pred_data.loc[pred_data[EyeTrackingFeatures.DATASET.value] == set_mode]
        # meta_data_panda_Iphone = pred_data_panda[pred_data_panda[EyeTrackingFeatures.DEVICE_NAME.value].str.contains('iPhone')]
        # meta_data_panda_Iphone.fillna(value='', inplace=True)
        # meta_dict = meta_data_panda_Iphone.to_dict('list')
        meta_dict = pred_data.to_dict('list')
        x_e = abs(np.subtract(meta_dict['label_dot_x_cam'], meta_dict['x_predict']))
        x_me = np.sum(x_e)/np.size(x_e)
        x_var = np.var(x_e)
        y_e = abs(np.subtract(meta_dict['label_dot_y_cam'], meta_dict['y_predict']))
        y_me = np.sum(y_e) / np.size(y_e)
        y_var = np.var(y_e)
        meta_dict['x_e'] = x_e
        meta_dict['y_e'] = y_e
        meta_panda = pd.DataFrame.from_dict(meta_dict)
        dict_for_table = {'device_name': [], 'x_me': [], 'x_variance':[], 'y_me': [], 'y_variance': [],
                          'x_screen_percentage_me': [], 'y_screen_percentage_me': []}
        for device_name in list(set(meta_panda['device_name'])):
            x_e_list = meta_panda[meta_panda['device_name'] == device_name]['x_e']
            x_device_me = np.sum(x_e_list)/np.size(x_e_list)
            x_device_var = np.var(x_e_list)
            # screen size is mm and me is cm
            x_screen_percentage_me = round((x_device_me/(((int(screen_size_panda[screen_size_panda['device'] == device_name]['width']))/int(screen_size_panda[screen_size_panda['device'] == device_name]['ppi']))*2.54))*100,2)
            dict_for_table['x_screen_percentage_me'].append(x_screen_percentage_me)
            meta_panda.loc[meta_panda['device_name'] == device_name, "x_device_me"] = x_device_me
            dict_for_table['device_name'].append(device_name)
            dict_for_table['x_me'].append(round(x_device_me,2))
            dict_for_table['x_variance'].append(round(x_device_var,2))
            y_e_list = meta_panda[meta_panda['device_name'] == device_name]['y_e']
            y_device_me = np.sum(y_e_list)/np.size(y_e_list)
            y_device_var = np.var(y_e_list)
            # screen size is mm and me is cm
            y_screen_percentage_me = round((y_device_me/((int(screen_size_panda[screen_size_panda['device'] == device_name]['hight'])/int(screen_size_panda[screen_size_panda['device'] == device_name]['ppi']))*2.54))*100,2)
            meta_panda.loc[meta_panda['device_name'] == device_name, "y_device_me"] = y_device_me
            meta_panda.loc[meta_panda['device_name'] == device_name, "y_screen_percentage_me"] = y_screen_percentage_me
            dict_for_table['y_screen_percentage_me'].append(y_screen_percentage_me)
            dict_for_table['y_me'].append(round(y_device_me,2))
            dict_for_table['y_variance'].append(round(y_device_var,2))
        dict_for_table['device_name'].append('all_devices')
        dict_for_table['x_me'].append(round(x_me,2))
        dict_for_table['x_variance'].append(round(x_var,2))
        dict_for_table['y_me'].append(round(y_me,2))
        dict_for_table['y_variance'].append(round(y_var,2))
        dict_for_table['y_screen_percentage_me'].append('irrelevant')
        dict_for_table['x_screen_percentage_me'].append('irrelevant')

        table_panda = pd.DataFrame.from_dict(dict_for_table)
        fig = ff.create_table(table_panda, height_constant=30)
        # Make text size larger
        for i in range(len(fig.layout.annotations)):
            fig.layout.annotations[i].font.size = 20
        fig.update_layout(title_text='filtered_batch32')
        fig.update_layout({'margin':{'t':50}})

        fig.show()

        print('X_ME:{} , Y_ME:{} , X_VAR'.format(x_me, y_me))

    def calc_prediction_by_users(self, set_mode: str, curr_device: str):
        pred_data = pd.read_csv(os.path.join(self._path_to_data, 'predicted_data.csv'))
        meta_dict = pred_data.to_dict('list')
        screen_size_panda = pd.read_csv(r'/mnt/2ef93ccf-c66e-4beb-95ba-24011e8fee18/TAMAR/screen_size.csv')
        x_e = abs(np.subtract(meta_dict['label_dot_x_cam'], meta_dict['x_predict']))
        x_me = np.sum(x_e) / np.size(x_e)
        x_var = np.var(x_e)
        y_e = abs(np.subtract(meta_dict['label_dot_y_cam'], meta_dict['y_predict']))
        y_me = np.sum(y_e) / np.size(y_e)
        y_var = np.var(y_e)
        meta_dict['x_e'] = x_e
        meta_dict['y_e'] = y_e
        meta_panda = pd.DataFrame.from_dict(meta_dict)
        dict_for_table = {'recording_id': [], 'device': [], 'x_me': [], 'x_variance': [], 'y_me': [], 'y_variance': []}
        for recording_id in list(set(meta_panda['recording_id'])):
            x_e_list = meta_panda[meta_panda['recording_id'] == recording_id]['x_e']
            x_device_me = np.sum(x_e_list) / np.size(x_e_list)
            x_device_var = np.var(x_e_list)
            meta_panda.loc[meta_panda['recording_id'] == recording_id, "x_recording_id_mse"] = x_device_me
            dict_for_table['recording_id'].append(recording_id)
            device = list(meta_panda[meta_panda['recording_id'] == recording_id]['device_name'])[0]
            normal_x = (int(screen_size_panda[screen_size_panda['device'] == device]['width'])/int(screen_size_panda[screen_size_panda['device'] == device]['ppi']))*2.54
            normal_y = (int(screen_size_panda[screen_size_panda['device'] == device]['hight'])/int(screen_size_panda[screen_size_panda['device'] == device]['ppi']))*2.54
            dict_for_table['device'].append(list(meta_panda[meta_panda['recording_id'] == recording_id]['device_name'])[0])
            # dict_for_table['x_me'].append(round(x_device_me/(normal_x)*100,2))
            # dict_for_table['x_variance'].append(round(x_device_var/(normal_x)*100,2))
            dict_for_table['x_me'].append(x_device_me)
            dict_for_table['x_variance'].append(round(x_device_var/(normal_x)*100,2))
            y_e_list = meta_panda[meta_panda['recording_id'] == recording_id]['y_e']
            y_device_me = np.sum(y_e_list) / np.size(y_e_list)
            y_device_var = np.var(np.sqrt(y_e_list))
            meta_panda.loc[meta_panda['recording_id'] == recording_id, "y_recording_id_me"] = y_device_me
            dict_for_table['y_me'].append(round(y_device_me/(normal_y)*100,2))
            dict_for_table['y_variance'].append(round(y_device_var/(normal_y)*100,2))
        dict_for_table['recording_id'].append('all_devices')
        dict_for_table['device'].append('irrelevant')
        dict_for_table['x_me'].append(x_me)
        dict_for_table['x_variance'].append(x_var)
        dict_for_table['y_me'].append(y_me)
        dict_for_table['y_variance'].append(y_var)
        table_panda = pd.DataFrame.from_dict(dict_for_table)
        histo_panda = table_panda[table_panda["device"] == curr_device]
        std = np.std(histo_panda['x_me'])
        x_tot_mse = np.mean(histo_panda['x_me'])
        # x_std_calc = []
        # for item in histo_panda['x_me']:
        #     if item < x_tot_mse + 0.25*std:
        #         x_std_calc.append(x_tot_mse + 0.25*std)
        #     elif item < x_tot_mse + 0.5*std:
        #         x_std_calc.append(x_tot_mse + 0.5*std)
        #     else:
        #         x_std_calc.append(x_tot_mse + std)
        # histo_panda['x_std_calc'] = x_std_calc
        fig = px.histogram(histo_panda, x="x_me")

        fig.show()

        # fig = px.scatter(table_panda, x="recording_id", y="x_me", color="device",
        #                  error_y="x_variance", error_y_minus="x_variance")
        # fig.update_layout(yaxis2=dict(range=[-50, 50]))
        #
        # fig.show()
        # fig = ff.create_table(table_panda, height_constant=30)
        # # Make text size larger
        # for i in range(len(fig.layout.annotations)):
        #     fig.layout.annotations[i].font.size = 20
        # fig.show()
        # polar_panda = pd.DataFrame.from_dict(dict_for_table)
        # fig = px.scatter_polar(polar_panda, r="x_me", theta="recording_id")
        #
        # fig.show()
        #
        # fig = px.bar_polar(polar_panda, r="x_me", theta="recording_id", color="x_variance",
        #                     color_discrete_sequence=px.colors.sequential.Plasma_r)
        # fig.show()
        # df = px.data.wind()
        # fig = px.line_polar(df, r="frequency", theta="direction", color="strength", line_close=True,
        #                     color_discrete_sequence=px.colors.sequential.Plasma_r,
        #                     template="plotly_dark",)
        # fig.show()

    # def calc_prediction_by_users_device(self, set_mode: str, device:str):


    def calc_hist_by_device(self, folder_path: str):
        all_models_df = pd.Series()
        for file in os.listdir(folder_path):
            curr_df = pd.read_csv(os.path.join(folder_path, file))
            curr_df['model'] = file.split('.')[0]
            if all_models_df.empty:
                all_models_df = curr_df

            else:
                all_models_df = all_models_df.append(curr_df)
        meta_dict = all_models_df.to_dict('list')
        x_e = abs(np.subtract(meta_dict['label_dot_x_cam'], meta_dict['x_predict']))
        x_me = np.sum(x_e)/np.size(x_e)
        x_var = np.var(x_e)
        y_e = abs(np.subtract(meta_dict['label_dot_y_cam'], meta_dict['y_predict']))
        y_me = np.sum(y_e) / np.size(y_e)
        y_var = np.var(y_e)
        meta_dict['x_me'] = x_e
        meta_dict['y_me'] = y_e
        meta_dict['x_std'] = x_e
        meta_dict['y_std'] = y_e
        meta_panda = pd.DataFrame.from_dict(meta_dict)
        x_error_panda = pd.DataFrame(meta_panda.groupby(['device_name', 'model'])["x_me"].mean())
        x_error_panda['model'] = [a[1] for a in x_error_panda.index]
        x_error_panda['device'] = [a[0] for a in x_error_panda.index]
        y_error_panda = pd.DataFrame(meta_panda.groupby(['device_name', 'model'])["y_me"].mean())
        y_error_panda['model'] = [a[1] for a in y_error_panda.index]
        y_error_panda['device'] = [a[0] for a in y_error_panda.index]
        x_std_panda = pd.DataFrame(meta_panda.groupby(['device_name', 'model'])["x_std"].std())
        x_std_panda['model'] = [a[1] for a in x_std_panda.index]
        x_std_panda['device'] = [a[0] for a in x_std_panda.index]
        y_std_panda = pd.DataFrame(meta_panda.groupby(['device_name', 'model'])["y_std"].std())
        y_std_panda['model'] = [a[1] for a in y_std_panda.index]
        y_std_panda['device'] = [a[0] for a in y_std_panda.index]
        fig = px.scatter(x_error_panda, x="device", y="x_me", color="model", symbol="model")
        fig.update_traces(marker_size=20)
        fig.update_xaxes(title_font = {"size": 20},tickfont=dict(family='Rockwell', size=20))
        fig.update_yaxes(title_font = {"size": 20},tickfont=dict(family='Rockwell', size=20))
        fig.show()
        fig = px.scatter(y_error_panda, x="device", y="y_me", color="model", symbol="model")
        fig.update_xaxes(title_font = {"size": 20},tickfont=dict(family='Rockwell', size=20))
        fig.update_yaxes(title_font = {"size": 20},tickfont=dict(family='Rockwell', size=20))
        fig.update_traces(marker_size=20)
        fig.show()
        fig = px.scatter(x_std_panda, x="device", y="x_std", color="model", symbol="model")
        fig.update_traces(marker_size=20)
        fig.update_xaxes(title_font = {"size": 20},tickfont=dict(family='Rockwell', size=20))
        fig.update_yaxes(title_font = {"size": 20},tickfont=dict(family='Rockwell', size=20))
        fig.show()
        fig = px.scatter(y_std_panda, x="device", y="y_std", color="model", symbol="model")
        fig.update_traces(marker_size=20)
        fig.update_xaxes(title_font = {"size": 20},tickfont=dict(family='Rockwell', size=20))
        fig.update_yaxes(title_font = {"size": 20},tickfont=dict(family='Rockwell', size=20))
        fig.show()

    def calc_2d_hist(self, folder_path:str):
        all_models_df = pd.Series()
        for file in os.listdir(folder_path):
            info_panda = pd.read_csv(os.path.join(folder_path, file))
            curr_df = info_panda[info_panda['dataset'] == 'test']
            curr_df['model'] = file.split('.')[0]
            curr_df['z_error'] = curr_df.apply(lambda row: np.sqrt(
                 (row.x_predict * row.x_predict) + (row.y_predict * row.y_predict)), axis=1)
            fig = px.density_heatmap(curr_df, x="label_dot_x_cam", y="label_dot_y_cam", z="z_error", histfunc="avg", title=file.split('.')[0])
            fig.show()
            #if all_models_df.empty:
                #all_models_df = curr_df
                # all_models_df['z'] = all_models_df.apply(lambda row: np.sqrt(
                #     (row.label_dot_x_cam * row.label_dot_x_cam) + (row.label_dot_y_cam * row.label_dot_y_cam)), axis=1)
                #fig = px.density_heatmap(curr_df, x="label_dot_x_cam", y="label_dot_y_cam", histfunc="count", title=curr_df['model'])
                #fig.show()

            #else:
                #continue

    def create_eye_crop_resolution_hist(self, path_to_eye_crops: str):
        meta_data = pd.read_csv(os.path.join(self._path_to_data, 'metadata.csv'))
        meta_data['eye_resolution'] = ""
        for i in meta_data.index:
            img = PIL.Image.open(meta_data.at[i, 'right_eye_frame_path'])
            wid, hgt = img.size
            resolution = str(wid) + "x" + str(hgt)
            meta_data.at[i, 'eye_resolution'] = resolution
        meta_data.sort_values('eye_resolution')
        fig = px.histogram(meta_data, x="eye_resolution")

        fig.show()


    def crate_filtered_prepared_data(self):
        meta_data = pd.read_csv(os.path.join(self._path_to_data, 'predicted_data.csv'))
        meta_data['error'] = ''

        for i in meta_data.index:
            error = np.sqrt(((meta_data.at[i, 'x_predict'])*(meta_data.at[i, 'x_predict']))+((meta_data.at[i, 'y_predict'])*(meta_data.at[i, 'y_predict'])))
            meta_data.at[i, 'error'] = error
        std = meta_data['error'].std()
        #filtered_df = meta_data[meta_data['error'] <= 3*std]
        #final_df = filtered_df[filtered_df['dataset'] == 'train']
        #save_df = final_df.drop(columns=['x_predict', 'y_predict'])
        #save_df.to_csv(r'/mnt/2ef93ccf-c66e-4beb-95ba-24011e8fee18/TAMAR/filtered_data/metadata.csv')
        bad_frames_panda = meta_data[meta_data['error'] > 3*std]
        bad_frames_panda.to_csv(r'/mnt/2ef93ccf-c66e-4beb-95ba-24011e8fee18/TAMAR/filtered_data/bad_frames.csv')





if __name__ == "__main__":
    #eye_tracker_analyser = EyeTrackerAnalyser(r'/mnt/2ef93ccf-c66e-4beb-95ba-24011e8fee18/TAMAR/filtered_data/normal_batch_32')
    #eye_tracker_analyser.calc_x_y_prediction_table('test')

    # eye_tracker_analyser = EyeTrackerAnalyser(r'/mnt/2ef93ccf-c66e-4beb-95ba-24011e8fee18/TAMAR/first_model_normal_batch_32')
    # eye_tracker_analyser.calc_prediction_by_users('test', "iPhone 5")

    # eye_tracker_analyser = EyeTrackerAnalyser(r'/mnt/2ef93ccf-c66e-4beb-95ba-24011e8fee18/TAMAR/prepared_data')
    # eye_tracker_analyser.analyse_data('test')

    #eye_tracker_analyser = EyeTrackerAnalyser(r'/mnt/2ef93ccf-c66e-4beb-95ba-24011e8fee18/TAMAR/prepared_data')
    #eye_tracker_analyser.calc_hist_by_device(r'/mnt/2ef93ccf-c66e-4beb-95ba-24011e8fee18/TAMAR/filtered_data/all_models')

    eye_tracker_analyser = EyeTrackerAnalyser(r'/mnt/2ef93ccf-c66e-4beb-95ba-24011e8fee18/TAMAR/prepared_data')
    eye_tracker_analyser.calc_2d_hist(r'/mnt/2ef93ccf-c66e-4beb-95ba-24011e8fee18/TAMAR/filtered_data/all_models')

    # eye_tracker_analyser = EyeTrackerAnalyser(r'/mnt/2ef93ccf-c66e-4beb-95ba-24011e8fee18/TAMAR/prepared_data')
    # eye_tracker_analyser.create_eye_crop_resolution_hist(r'/mnt/2ef93ccf-c66e-4beb-95ba-24011e8fee18/TAMAR/prepared_data')

    #eye_tracker_analyser = EyeTrackerAnalyser(r'/mnt/2ef93ccf-c66e-4beb-95ba-24011e8fee18/TAMAR/first_model_normal_batch_32')
    #eye_tracker_analyser.crate_filtered_prepared_data()