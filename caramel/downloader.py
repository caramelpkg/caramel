import os
from pathlib import Path


def _charconv1(char):
    return("'{}'".format(char))


def _charconv2(char):
    return('"{}"'.format(char))


class downloader(object):
    def __init__(self, latll: float, latur: float, lonll: float, lonur: float, datestart: str, dateend: str):
        '''
            Initialize the downloader object
            Input:
                latll [float]: a latitude at the lower left corner of the region of interest           
                latur [float]: a latitude at the upper right corner of the region of interest
                lonll [float]: a longitude at the lower left corner of the region of interest
                lonur [float]: a longitude at the upper right corner of the region of interst
                datestart [str]: the start date in "YYYY-MM-DD"
                datesend  [str]: the end date in "YYYY-MM-DD"
        '''
        self.latll = latll
        self.latur = latur
        self.lonll = lonll
        self.lonur = lonur
        self.datestart = datestart
        self.dateend = dateend

    def download_tropomi_l2(self, product: str, output_fld: Path, maxpage=30, username="s5pguest", password="s5pguest"):
        '''
            download the tropomi data
            Inputs:
                product [int]: 1 -> NO2
                               2 -> HCHO
                               3 -> CH4
                               4 -> CO
                output_fld [Path]: a pathlib object describing the output folder
                maxpage [int]: the number of pages in the xml file
                username [str]: the username to log on s5phub
                password [str]: the password to log on s5phub 
        '''
        # define product string
        if product == 1:
            product_name = "%5F%5FNO2%5F%5F%5F"
        if product == 2:
            product_name = "%5F%5FHCHO%5F%5F"
        if product == 3:
            product_name = "%5F%5FCH4%5F%5F%5F"
        if product == 4:
            product_name = "%5F%5FCO%5F%5F%5F%5F"
        # loop over the pages
        for page in range(0, maxpage):
            searcher = "https://s5phub.copernicus.eu/dhus/search?start="
            searcher += f"{0+page*100:01}" + "&rows=100&q=footprint:%22"
            searcher += "Intersects(POLYGON((" + f"{self.lonll:.4f}" + "%20"
            searcher += f"{self.latll:.4f}" + "," + f"{self.lonur:.4f}" + "%20"
            searcher += f"{self.latll:.4f}" + "," + \
                f"{self.lonur:.4f}" + "%20" + f"{self.latur:.4f}"
            searcher += "," + f"{self.lonll:.4f}" + \
                "%20" + f"{self.latur:.4f}" + ","
            searcher += f"{self.lonll:.4f}" + "%20" f"{self.latll:.4f}" + \
                ")))%22%20AND%20(%20beginPosition:"
            searcher += "%5B" + self.datestart + "T00:00:00.000Z%20TO%20" + self.dateend
            searcher += "T23:59:59.999Z%5D%20AND%20endPosition:%5B" + self.datestart
            searcher += "T00:00:00.000Z%20TO%20" + self.dateend + "T23:59:59.999Z%5D%20)%"
            searcher += "20AND%20((platformname:Sentinel-5)%20AND%20(producttype:"
            searcher += "L2"f"{product_name}""%20AND%20processinglevel:L2))"

            # retrieve the product names and save them in temp/tropomi.xml
            if not os.path.exists('temp'):
                os.makedirs('temp')
            cmd = ' curl -L -k -u s5pguest:s5pguest "' + searcher + '"> temp/tropomi.xml'
            os.system(cmd)

            # read the xml file
            with open('temp/tropomi.xml', 'r') as f:
                data = f.read()
                data1 = data.split('<str name="uuid">')

                # list the files to be downloaded in this particular page
                list_file = []

                if len(data1) == 0:
                    break

                for i in range(0, len(data1)-1):
                    list_file.append(data1[i+1].split('</str>')[0])

                # download the data
                for fname in list_file:
                    cmd = "wget -nH -nc --no-check-certificate --content-disposition --continue "
                    cmd += "--user s5pguest --password s5pguest "
                    cmd += _charconv2("https://s5phub.copernicus.eu/dhus/odata/v1/Products(" +
                                      _charconv1(fname) + ")/\$value")
                    cmd += " -P" + (output_fld.as_posix()) + "/"
                    if not os.path.exists(output_fld.as_posix()):
                        os.makedirs(output_fld.as_posix())
                    os.system(cmd)


# testing
if __name__ == "__main__":

    dl_obj = downloader(37, 40, -79, -73.97, '2020-06-01', '2020-06-10')
    dl_obj.download_tropomi_l2(2, Path('download_bucket/hcho/'))
