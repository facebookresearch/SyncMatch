# flake8: noqa
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Python to HTML writer
Based on prior version from 
- Rohit Girdhar (https://github.com/rohitgirdhar/PyHTMLWriter)
- Richard Higgins (Edited original version)
"""
import csv
import errno
import inspect
import io
import math
import os
import random
import string
import uuid
from urllib import request as urllib

from PIL import Image


class Element:
    """ A data element of a row in a table """

    def __init__(self, htmlCode="", drawBorderColor=""):
        self.htmlCode = htmlCode
        self.isHeader = False
        self.drawBorderColor = drawBorderColor

    def textToHTML(self, text):
        res = "<p><b>" + text + "</b></p>"
        return res

    def imgToHTML(self, img_path, width=300, overlay_path=None):
        res = '<img data-src="' + img_path.strip().lstrip() + '" '
        res += f'style="height: {width}px" '
        if overlay_path:
            res += "ondblclick=\"this.src='" + overlay_path.strip().lstrip() + "';\""
            res += "onmouseout=\"this.src='" + img_path.strip().lstrip() + "';\""
        res += "/>"
        return res

    def vidToHTML(self, vid_path, width=320):
        vid_type = "mp4"
        res = """
            <video width="%d" controls>
                <source src="%s" type="video/%s">
                Your browser does not support the video tag.
            </video>""" % (
            width,
            vid_path,
            vid_type,
        )
        return res

    def imgToBboxHTML(
        self, img_path, bboxes, col="green", wid=300, ht=300, imsize=None
    ):
        idd = "img_" + "".join(
            random.SystemRandom().choice(string.ascii_uppercase + string.digits)
            for _ in range(10)
        )

        # compute the ratios
        if imsize:
            actW = imsize[0]
            actH = imsize[1]
        else:
            actW, actH = self.tryComputeImgDim(img_path)
        actW = float(actW)

        actH = float(actH)
        if actW > actH:
            ht = wid * (actH / actW)
        else:
            wid = ht * (actW / actH)
        ratioX = wid / actW
        ratioY = ht / actH

        for i in range(len(bboxes)):
            bboxes[i] = [
                bboxes[i][0] * ratioX,
                bboxes[i][1] * ratioY,
                bboxes[i][2] * ratioX,
                bboxes[i][3] * ratioY,
            ]
        colStr = ""
        if self.drawBorderColor:
            col = self.drawBorderColor
            colStr = "border: 10px solid " + col + ";"
        htmlCode = (
            """
            <canvas id="""
            + idd
            + """ style="border:1px solid #d3d3d3; """
            + colStr
            + """
                background-image: url("""
            + img_path
            + """);
                background-repeat: no-repeat;
                background-size: contain;"
                width="""
            + str(wid)
            + """,
                height="""
            + str(ht)
            + """>
           </canvas>
           <script>
                var c = document.getElementById(\""""
            + idd
            + """\");
                var ctx = c.getContext("2d");
                ctx.lineWidth="2";
                ctx.strokeStyle=\""""
            + col
            + """\";"""
        )
        for i in range(len(bboxes)):
            htmlCode += (
                """ctx.rect(""" + ",".join([str(i) for i in bboxes[i]]) + """);"""
            )
        htmlCode += """ctx.stroke();
        </script>
        """
        return htmlCode

    def addImg(self, img_path, **kwargs):
        self.htmlCode += self.imgToHTML_base(img_path, **kwargs)

    def imgToHTML_base(
        self,
        img_path,
        width=500,
        bboxes=None,
        imsize=None,
        overlay_path=None,
        poses=None,
        scale=None,
    ):
        # bboxes must be a list of [x,y,w,h] (i.e. a list of lists)
        # imsize is the natural size of image at img_path.. used for putting bboxes, not required otherwise
        # even if it's not provided, I'll try to figure it out -- using the typical use cases of this software
        # overlay_path is image I want to show on mouseover
        if bboxes:
            # TODO overlay path not implemented yet for canvas image
            return self.imgToBboxHTML(img_path, bboxes, "green", width, width, imsize)
        elif poses:
            return self.imgToPosesHTML(
                img_path, poses, width, width, imsize, overlay_path
            )
        else:
            return self.imgToHTML(img_path, width, overlay_path)

    def addVideo(self, vid_path):
        self.htmlCode += self.vidToHTML(vid_path)

    def addTxt(self, txt):
        if self.htmlCode:  # not empty
            self.htmlCode += "<br />"
        self.htmlCode += str(txt)

    def addLink(self, url, name=None):
        if name is not None:
            self.htmlCode = f'<a href="{url}">{name}</a>'
        else:
            self.htmlCode = f'<a href="{url}">{url}</a>'

    def getHTML(self):
        return self.htmlCode

    def setIsHeader(self):
        self.isHeader = True

    def setDrawCheck(self):
        self.drawBorderColor = "green"

    def setDrawUnCheck(self):
        self.drawBorderColor = "red"

    def setDrawBorderColor(self, color):
        self.drawBorderColor = color

    @staticmethod
    def getImSize(impath):
        im = Image.open(impath)
        return im.size

    @staticmethod
    def tryComputeImgDim(impath):
        try:
            im = Image.open(impath)
            res = im.size
            return res
        except:
            pass
        try:
            # most HACKY way to do this, remove the first '../'
            # since most cases
            impath2 = impath[3:]
            return self.getImSize(impath2)
        except:
            pass
        try:
            # read from internet
            fd = urllib.urlopen(impath)
            image_file = io.BytesIO(fd.read())
            im = Image.open(image_file)
            return im.size
        except:
            pass
        print("COULDNT READ THE IMAGE SIZE!")


class Table:
    def __init__(self, rows=[], path=None):
        self.path = path
        self.rows = [row for row in rows if not row.isHeader]
        self.headerRows = [row for row in rows if row.isHeader]

    def addRow(self, row):
        if not row.isHeader:
            self.rows.append(row)
        else:
            self.headerRows.append(row)

    def getHTML(
        self,
        makeChart=False,
        transposeTableForChart=False,
        chartType="line",
        chartHeight=650,
    ):
        html = '<table border=1 id="data" class="sortable">'
        for r in self.headerRows + self.rows:
            html += r.getHTML()
        html += "</table>"
        if makeChart:
            html += self.genChart(
                transposeTable=transposeTableForChart,
                chartType=chartType,
                chartHeight=chartHeight,
            )
        return html

    def readFromCSV(self, fpath, scale=1.0):
        with open(fpath) as f:
            tablereader = csv.reader(filter(lambda row: row[0] != "#", f))
            for row in tablereader:
                tr = TableRow()
                for elt in row:
                    try:
                        tr.addElement(Element(str(float(elt) * scale)))
                    except:
                        tr.addElement(Element(elt))
                self.addRow(tr)

    def countRows(self):
        return len(self.rows)

    def genChart(self, transposeTable=False, chartType="line", chartHeight=650):
        # Generate HighCharts.com chart using the table
        # data. Assumes that data is numeric, and first row
        # and the first column are headers
        for row in self.rows:
            row.elements[0].setIsHeader()
        scrdir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        )
        f = open(os.path.join(scrdir, "../templates/highchart_js.html"))
        base_js = f.read()
        f.close()
        base_js = string.Template(base_js).safe_substitute(
            {"transpose": "true"} if transposeTable else {"transpose": "false"}
        )
        base_js = string.Template(base_js).safe_substitute(
            {"chartType": "'" + chartType + "'"}
        )
        base_js = string.Template(base_js).safe_substitute(
            {"chartHeight": str(chartHeight)}
        )
        return base_js


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


class TableRow:
    def __init__(self, isHeader=False, rno=-1, elementsPerRow=9999999999):
        self.isHeader = isHeader
        self.elements = []
        self.rno = rno
        self.elementsPerRow = elementsPerRow

    def addElement(self, element):
        self.elements.append(element)

    def getHTML(self):
        html = ""
        for elements in chunks(self.elements, self.elementsPerRow):
            html += "<tr>"
            if self.rno >= 0:
                html += '<td><a href="#' + str(self.rno) + '">' + str(self.rno) + "</a>"
                html += "<a name=" + str(self.rno) + "></a></td>"
            for e in elements:
                if self.isHeader or e.isHeader:
                    elTag = "th"
                else:
                    elTag = "td"
                html += "<%s>" % elTag + e.getHTML() + "</%s>" % elTag
            html += "</tr>\n"
        return html


class TableWriter:
    def __init__(
        self,
        table,
        rowsPerPage=20,
        pgListBreak=20,
        makeChart=False,
        topName="index",
        head="",
        desc="",
        transposeTableForChart=False,
        chartType="line",
        chartHeight=650,
    ):
        self.outputdir = table.path
        self.rowsPerPage = rowsPerPage
        self.table = table
        self.pgListBreak = pgListBreak
        self.makeChart = makeChart
        self.topName = topName
        self.desc = desc
        self.head = head
        self.transposeTableForChart = transposeTableForChart  # used in genCharts
        self.chartType = chartType  # used in genCharts
        self.chartHeight = chartHeight

    def write(self, writePgLinks=True):
        # returns a list with each element as (link to table
        # row, row)
        ret_data = []
        self.mkdir_p(self.outputdir)
        nRows = self.table.countRows()
        pgCounter = 1
        for i in range(0, nRows, self.rowsPerPage):
            rowsSubset = self.table.rows[i : i + self.rowsPerPage]
            t = Table(self.table.headerRows + rowsSubset)
            ret_data.append((pgCounter, rowsSubset))
            f = open(
                os.path.join(self.outputdir, f"{self.topName}{pgCounter:03d}.html"), "w"
            )

            f.write(
                """<head>
            <script src="http://www.kryogenix.org/code/browser/sorttable/sorttable.js"></script>
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.1.1/jquery.min.js"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/js/bootstrap.min.js"></script>
            <script>
            $(".ui-tabs-anchor").click(function(event) {
                $("div#"+this.getAttribute('href')).show();
                $("div#"+this.getAttribute('href')).siblings().not('.something').hide();
            });
            </script>
            <script src="https://cdn.jsdelivr.net/npm/vanilla-lazyload@12.0.0/dist/lazyload.min.js"></script>
            """
            )
            f.write(f"  <h1> {self.head} </h1>")
            f.write("</head>")
            f.write('<div align="left" class="parent">')
            f.write(self.desc)
            pgLinks = self.getPageLinks(
                int(math.ceil(nRows * 1.0 / self.rowsPerPage)),
                pgCounter,
                self.pgListBreak,
                self.topName,
            )
            if writePgLinks:
                f.write(pgLinks)
            f.write(
                t.getHTML(
                    makeChart=self.makeChart,
                    transposeTableForChart=self.transposeTableForChart,
                    chartType=self.chartType,
                    chartHeight=self.chartHeight,
                )
            )
            if writePgLinks:
                f.write(pgLinks)
            f.write("</div>")
            f.write(self.getCredits())
            f.write(
                "<script>var LazyLoadInstance = new LazyLoad();</script></body></html>"
            )
            f.close()
            pgCounter += 1

        return ret_data

    @staticmethod
    def mkdir_p(path):
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise

    @staticmethod
    def getPageLinks(nPages, curPage, pgListBreak, topName):
        if nPages < 2:
            return ""
        links = ""
        for i in range(1, nPages + 1):
            if i != curPage:
                links += (
                    '<a href="'
                    + f"{topName}{i:03d}"
                    + '.html">'
                    + str(topName)
                    + str(i)
                    + "</a>&nbsp"
                )
            else:
                links += str(i) + "&nbsp"
            if i % pgListBreak == 0:
                links += "<br />"
        return "\n" + links + "\n"

    @staticmethod
    def getCredits():
        return '\n<br/><div align="center"><small>Generated using <a href="https://github.com/rohitgirdhar/PyHTMLWriter">PyHTMLWriter</a></small></div>'


def imgElement(image_path, width, alt_image_path=None):
    ele = Element()
    ele.addImg(image_path, width=width, overlay_path=alt_image_path)
    return ele


def vidElement(path):
    ele = Element()
    ele.addVideo(path)
    return ele
