import socks
import socket

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

if __name__ == '__main__':
    socks.set_default_proxy(socks.PROXY_TYPE_HTTP, "127.0.0.1", 7890)
    socket.socket = socks.socksocket

    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    # Set the ID of Google Drive folder needed to be downloaded
    parent_folder_id = '1r5sPiBEvo8Xby-nMaWUTnJIPK6WhY1B6'

    # Set the path of the local folder where the files will be downloaded
    parent_folder_dir = '.'

    if parent_folder_dir[-1] != '/':
        parent_folder_dir = parent_folder_dir + '/'

    # use gdown to download
    gdown_cmd = '[ ! -f {FILE_NAME} ] && gdown --no-cookies https://drive.google.com/uc?id={FILE_ID} -O {FILE_NAME}'
    md5_cmd = '{MD5_SUM} {FILE_NAME}'

    # Get the folder structure
    file_dict = dict()
    folder_queue = [parent_folder_id]
    dir_queue = [parent_folder_dir]
    cnt = 0

    while len(folder_queue) != 0:
        current_folder_id = folder_queue.pop(0)
        file_list = drive.ListFile({'q': f"'{current_folder_id}' in parents and trashed=false"}).GetList()

        current_parent = dir_queue.pop(0)
        print(current_parent, current_folder_id)
        for file_item in file_list:
            file_dict[cnt] = dict()
            file_dict[cnt]['id'] = file_item['id']
            file_dict[cnt]['title'] = file_item['title']
            file_dict[cnt]['dir'] = current_parent + file_item['title']

            if file_item['mimeType'] == 'application/vnd.google-apps.folder':
                file_dict[cnt]['type'] = 'folder'
                file_dict[cnt]['dir'] += '/'
                folder_queue.append(file_item['id'])
                dir_queue.append(file_dict[cnt]['dir'])
            else:
                file_dict[cnt]['type'] = 'file'
                file_dict[cnt]['md5Checksum'] = file_item['md5Checksum']

            cnt += 1

    f_script = open('script.sh', 'w')  # output script
    f_md5 = open('script.md5', 'w')  # output md5
    for file_iter in file_dict.keys():
        if file_dict[file_iter]['type'] == 'folder':
            f_script.write('mkdir ' + file_dict[file_iter]['dir'] + '\n')
        else:
            f_script.write(gdown_cmd.format(FILE_ID=file_dict[file_iter]['id'],FILE_NAME=file_dict[file_iter]['dir']) + '\n')
            f_md5.write(md5_cmd.format(MD5_SUM=file_dict[file_iter]['md5Checksum'],FILE_NAME=file_dict[file_iter]['dir']) + '\n')
    f_script.close()
    f_md5.close()
