import cx_Oracle
import sqlite3
import random
import reset


def insert_error(path_ori,path,error_rate):

    #conn = cx_Oracle.connect('system', 'Pjfpjf11', '127.0.0.1:1521/orcl')  # Connecting to the database
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    random.seed(1)
    path2 = path + "1"
    path3 = path + "2"
    reset.reset(path_ori,path)   # First, reset the
    # reset.reset(path_ori, path2)

    sql = "DELETE FROM \"" + path2 + "\" "
    cursor.execute(sql)
    conn.commit()  # 清空

    sql = "DELETE FROM \"" + path3 + "\" "
    cursor.execute(sql)
    conn.commit()  # 清空

    sql1 = "select * from \"" + path + "\" "
    print(sql1)
    cursor.execute(sql1)
    data1 = cursor.fetchall()  # All data
    des = cursor.description
    # print("Description of the table:", des)
    # print("Table Header:", ",".join([item[0] for item in des]))
    t1 = len(data1)   # Total data -1 can be changed to index position
    t2 = len(data1[0]) - 1  # Length of data per row, -1 is to remove the label column

    att_name=[]
    for item in des:
        att_name.append(item[0])
    # print(att_name)
    dict={}
    for i in range(t2):    #
        dict[i]=att_name[i]
    print(dict)
    # f = open('att_name.txt', 'w')
    f = open('data/save/att_name.txt', 'w')
    f.write(str(dict))
    f.close()

    # f = open('data/save/att_name.txt', 'r')
    # label2att = eval(f.read())
    # f.close()
    # print(label2att)

    att = list(dict.values())

    count_error_all=int(error_rate*t1)
    # print(l2)

    # count_error_MV= int(count_error_all/3)      #missing value
    # count_error_FI= int(count_error_all/3)      #formatting issue
    # count_error_typo=count_error_all-count_error_MV-count_error_FI


    error_index=random.sample(range(0,t1),count_error_all)    #Random 10 different numbers out of 10,000 numbers from 0-9999
    print(len(error_index))
    # print(error_index)
    count=0
    for index in error_index:
        row=data1[index]
        # print(row)

        for i in range(t2):  # t2
            if i == 0:
                sql_inf = f"\"{att[i]}\"='{row[i]}'"
            else:
                sql_inf += f" and \"{att[i]}\"='{row[i]}'"
        sql_info = sql_inf + " and \"Label\"='None'"
        # print(sql_info)
        if count<int(count_error_all/3):
            error_list = ["error1", "error2", "error3", "error4", "error5"]
            r = random.randint(1, t2 - 1)  # Assume a total of 9 columns, -1 divided by the last column label when calculating t2, now -1 again is to become the index position, so the number of random in 1-7
            r2 = random.randint(0, 4)
            error = error_list[r2]
        elif count<int(2*count_error_all/3):
            r = random.randint(1, t2 - 1)
            error="missing"                 # Note, here you can write error=" " as the missing value, but the execution of the model detection will replace all null values with "missing", so here it is written directly as missing
        else:
            r = random.randint(1, t2 - 1)
            sql = "select distinct (\"" + att[r] + "\") from \"" + path + "\""
            # print(sql)
            cursor.execute(sql)
            values = cursor.fetchall()
            error_value = random.choice(values) # Other values of the same column
            error=error_value[0]
            while (error==row[r]):
                error_value = random.choice(values)  # Other values of the same column
                error = error_value[0]
                # print(att[r], error, row[r])
        # print(r)
        # print(error)
        if (error is None):
            error=""
        sql2 = f"update \"{path}\" set \"Label\"='1' , \"{att[r]}\"='{error}' where {sql_info}" #and \"" + att[r] + "\"='error'
        # print(sql2)
        cursor.execute(sql2)
        conn.commit()



        # Generate Hosp_rules_copy2
        sql = "select * from \"" + path_ori + "\" where  " + sql_inf + ""
        cursor.execute(sql)
        data_clean = cursor.fetchall()
        # print(sql)
        # print(data_clean)
        t3 = len(data_clean[0])  # Length of data per row
        for num in range(t3):
            if num == 0:
                sql_before = "'%s'"
            else:
                sql_before = sql_before + ",'%s'"
        # print(sql_before)
        va = []
        for num in range(t3):
            va.append(data_clean[0][num])
        sql_after = tuple(va)
        sql_clean = "insert into \"" + path3 + "\" values(" + sql_before + ")"
        sql3=sql_clean% (sql_after)
        # print(sql4)
        cursor.execute(sql3)
        conn.commit()   # Reset

        # 生成Hosp_rules_copy1
        sql4 = "insert into \"" + path2 + "\" values(" + sql_before + ")"
        sql5 = sql4 % (sql_after)
        # print(sql4)
        cursor.execute(sql5)
        conn.commit()
        sql_dirty = f"update \"{path2}\" set \"Label\"='1' , \"{att[r]}\"='{error}' where {sql_inf}"
        cursor.execute(sql_dirty)
        conn.commit()
        count = count + 1



    sql_check = "select * from \"" + path + "\" where \"Label\"='1'"
    cursor.execute(sql_check)
    data2 = cursor.fetchall()
    print("Number of Generated Errors:", len(data2))    # Generating an erroneous redundancy expectation may be due to the presence of duplicate items in the data set

    cursor.close()
    conn.close()


if __name__ == '__main__':
    path_ori="Hosp_rules"
    path = "Hosp_rules_copy"
    # path_ori = "Food"
    # path = "Food_copy"
    # path_ori = "Gov_data"
    # path = "Gov_data_copy"
    # path_ori = "Flight"
    # path = "Flight_copy"

    error_rate = 0.1
    insert_error(path_ori,path,error_rate)
    print("Insert error complete")


