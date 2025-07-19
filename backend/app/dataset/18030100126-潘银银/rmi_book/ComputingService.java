
import java.util.*;
import java.io.*;
import java.rmi.RemoteException;
public interface ComputingService extends java.rmi.Remote {
	public void initBook() throws RemoteException, FileNotFoundException;
	public boolean add(Book b) throws RemoteException, FileNotFoundException;
	public boolean delete(int bookID) throws RemoteException, FileNotFoundException;
	public Book queryByID(int bookID) throws RemoteException;
	public BookList queryByName(String name) throws RemoteException;
	public String booksInfo() throws RemoteException;
	public void showAll() throws RemoteException;
	public void savetxt() throws RemoteException, FileNotFoundException;
}