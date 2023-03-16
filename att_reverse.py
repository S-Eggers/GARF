import cx_Oracle

def att_reverse(path,order):


    conn = cx_Oracle.connect('system', 'Pjfpjf11', '127.0.0.1:1521/orcl')  # Connecting to the database
    cursor = conn.cursor()
    sql1 = "select * from \"" + path + "\" "
    print(sql1)
    cursor.execute(sql1)
    data1 = cursor.fetchall()  # All data
    data1 = [x[:-1] for x in data1]     # Data with label removed
    if order == 0:
        data1 = [x[::-1] for x in data1]    # Reverse order
        des = cursor.description
        des.reverse()
        print(des)
        del (des[0])    # Remove the label from the table header
    else:
        des = cursor.description
    print(data1)

    # print(type(des))
    # print("Description of the table:", des)
    # print("Table Header:", ",".join([item[0] for item in des]))
    t1 = len(data1)  # Total data volume
    t2 = len(data1[0])  # Length of data per row

    att_name = []
    for item in des:
        # print(item)
        att_name.append(item[0])
    # print(att_name)
    dict = {}
    for i in range(t2):
        dict[i] = att_name[i]
    print(dict)
    # f = open('att_name.txt', 'w')
    f = open('data/save/att_name.txt', 'w')
    f.write(str(dict))
    f.close()


if __name__ == '__main__':


    path = "Hosp_rules_copy"

    att_reverse(path,1)
    # dict_generator()

