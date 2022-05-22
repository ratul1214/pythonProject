import requests
from bs4 import BeautifulSoup


# user define function
# Scrape the data
# and get in string
def getdata(url):
    r = requests.get(url)
    return r.text
def job_post_text(url):


    soup = html_code(url)

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()  # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    print(lines)

# Get Html code using parse
def html_code(url):
    # pass the url
    # into getdata function
    htmldata = getdata(url)
    soup = BeautifulSoup(htmldata, 'html.parser')

    # return html code
    return (soup)


# filter job data using
# find_all function
def job_data(soup):
    # find the Html tag
    # with find()
    # and convert into string
    data_str = ""
    for item in soup.find_all("a", class_="job-title-text turnstileLink"):
        data_str = data_str + item.get_text()
    result_1 = data_str.split("\n")
    return (result_1)


# filter company_data using
# find_all function


def company_data(soup):
    # find the Html tag
    # with find()
    # and convert into string
    data_str = ""
    result = ""
    for item in soup.find_all("div", class_="sjcl"):
        data_str = data_str + item.get_text()
    result_1 = data_str.split("\n")

    res = []
    for i in range(1, len(result_1)):
        if len(result_1[i]) > 1:
            res.append(result_1[i])
    return (res)
def returnJobLnk(soup):
    data_str = ""
    linkListBd = []
    print(soup.find_all("div", class_="job-title-text"))
    for item in soup.find_all("div", class_="job-title-text"):
        data_str = item.find('a',href = True)
        print(data_str['href'])
        #jobeDetails = html_code('https://jobs.bdjobs.com/'+data_str['href'])
        linkListBd.append('https://jobs.bdjobs.com/'+data_str['href'])
        job_post_text('https://jobs.bdjobs.com/'+data_str['href'])
    return linkListBd
def bdjob_scraping():
    bdjobSoup = html_code('https://jobs.bdjobs.com/jobsearch.asp?fcatId=8&icatId=')
    linkListBd =returnJobLnk(bdjobSoup)
    print(linkListBd)
# driver nodes/main function
if __name__ == "__main__":

    bdjob_scraping()
    # job = "data+science+internship"
    # Location = "Noida%2C+Uttar+Pradesh"
    # url = "https://in.indeed.com/jobs?q=" + job + "&l=" + Location
    #
    # # Pass this URL into the soup
    # # which will return
    # # html string
    # soup = html_code(url)
    #
    # # call job and company data
    # # and store into it var
    # job_res = job_data(soup)
    # com_res = company_data(soup)

    # # Traverse the both data
    # temp = 0
    # for i in range(1, len(job_res)):
    #     j = temp
    #     for j in range(temp, 2 + temp):
    #         print("Company Name and Address : " + com_res[j])
    #
    #     temp = j
    #     print("Job : " + job_res[i])
    #     print("-----------------------------")
