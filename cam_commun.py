import requests
from requests.auth import HTTPDigestAuth


class CameraConnection:
    def __init__(self, cam_ip = '192.168.10.225', cam_pswd='cap5241', cam_user_name='admin', cam_port=80) -> None:
        self.auth = HTTPDigestAuth(cam_user_name, cam_pswd) if cam_user_name and cam_pswd else None
        self.min_zoom = 1.0
        self.max_zoom = 42.0
        self.cam_ip = cam_ip

    def send_request(self, url) -> requests.Response | None:
        try:
            if self.auth:
                response = requests.get(url, auth=self.auth)
                return response
            else:
                print('no auth, cant send request')
                # response = requests.get(url)
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    @staticmethod
    def check_response(response) -> bool:
        if response is None:
            print("None response")
            return False
        #Check the response from the camera and handle errors.
        elif response.status_code == 200:
            response_text = response.text.strip()  # Strip any leading/trailing whitespace
            if response_text.startwith("Error"):
                return True
            else:
                print(f"Configuration failed: {response_text}")
                return False
        else:
            print(f"Request failed with status code: {response.status_code}")
            return False
        
    def set_zoom(self, data):
        if data < self.min_zoom or data > self.max_zoom:
            # print('invalid data value: ', data)
            return False, False
        data = data * 100 #zoom value to url is in range 100-4200 
        req_url = f'http://{self.cam_ip}/cgi-bin/ptz.cgi?action=start&channel=1&code=PositionABSHDX&arg1=0&arg2=0&arg3={data}'
        response = self.send_request(req_url)
        return self.check_response(response), True
    
    def cont_zoom_in(self, data):
        #data is either start or stop
        req_url = f'http://{self.cam_ip}/cgi-bin/ptz.cgi?action={data}&channel=1&code=ZoomTele&arg1=0&arg2=0&arg3=0'
        response = self.send_request(req_url)
        print(f'response: \n{response.text}')
        return self.check_response(response), True
    
    def cont_zoom_out(self, data):
        req_url = f'http://{self.cam_ip}/cgi-bin/ptz.cgi?action={data}&channel=1&code=ZoomWide&arg1=0&arg2=0&arg3=0'
        response = self.send_request(req_url)
        return self.check_response(response), True