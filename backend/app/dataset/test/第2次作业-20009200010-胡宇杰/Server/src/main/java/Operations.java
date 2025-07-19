import java.rmi.Remote;
import java.rmi.RemoteException;
import java.util.ArrayList;

//接口设计部分
public interface Operations extends Remote{
    boolean add(Book b) throws RemoteException;
    Book queryByID(int bookID) throws RemoteException;
    ArrayList queryByName(String name) throws RemoteException;
    boolean delete(int bookID) throws RemoteException;
    boolean ListAll() throws RemoteException;//在服务器端列出所有的图书，便于判断程序是否运行正确
}
