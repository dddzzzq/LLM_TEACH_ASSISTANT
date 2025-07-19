import java.rmi.RemoteException;
import java.util.ArrayList;

//接口实现部分
public class Operationslmpl implements Operations{
    public static Book[] allbook=new Book[20];
    public static int i=0;
    //public static int j=0;

    public boolean add(Book book) throws RemoteException {
        System.out.println("Someone is calling me. id : " + book.GetId() + "   name=" + book.GetName());
        allbook[i++]=book;//将对应图书放入数组中
        return true;
    }
    public Book queryByID(int bookID) throws RemoteException{
        Book book=new Book();
        int j=0;
        for(;j<i&&allbook[j].GetId()!=bookID;j++);       //遍历得到对应编号的书籍

        if(j<i){                                         //存在对应书籍
            System.out.println("Get it! QueryByID success!");
            book=allbook[j];
        }
        else{                                            //不存在对应书籍
            System.out.println("fail to query.The ID is wrong.");
        }
        return book;
    }

    public ArrayList queryByName(String name) throws RemoteException{
        System.out.println("Someone is calling me. in function queryByName");
        ArrayList bookList=new ArrayList<>();
        for(int j=0;j<i;j++){
            if(allbook[j].GetName().contains(name)){
                bookList.add(allbook[j]);
            }
        }
        return bookList;
    }
    public boolean delete(int bookID) throws RemoteException{
        int j=0;
        for(;j<i&&allbook[j].GetId()!=bookID;j++);      //遍历得到对应编号的书籍
        //删除该书籍
        if(j<(i--)){
            System.out.println("the book 《"+allbook[j].GetName()+"》 has been deleted.");
            for(;j<i;j++) allbook[j]=allbook[j+1];
            return true;
        }
        else{
            System.out.println("delete error! The bookID is wrong.");
            return false;
        }
    }
    public boolean ListAll() throws RemoteException{
        System.out.println("图书列表如下：");
        for(int j=0;j<i;j++){
            System.out.println("name : "+allbook[j].GetName()+"  id : "+allbook[j].GetId());
        }
        return true;
    }
}
