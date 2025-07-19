import java.util.*;
import java.io.*;
//import java.io.Serializable;
import java.rmi.RemoteException;


class Book implements Serializable {
	private int bookID;
	private String bookName;
    public Book(int id, String name){
		this.bookID = id;
		this.bookName = name;
    }
	public int getID() {
		return bookID;
	}
	public String getName() {
		return bookName;
	}
	public String getInfo() {
		return  "ID: " + bookID + "  name:" + bookName + "\n";
	}
	public void showInfo() {
		System.out.println(getInfo());
	}
}


class BookList implements Serializable {
	ArrayList<Book> booklist = new ArrayList<Book>();
	public String getInfo() {
		String info ="";
		for (int i = 0; i < booklist.size(); i++) {
			info += ("id:" + booklist.get(i).getID() + " name:" + booklist.get(i).getName() + "\n");
		}
		return info;
	}
	public void showInfo() {
		System.out.println(getInfo());
	}
}
class ComputingServiceImpl extends java.rmi.server.UnicastRemoteObject implements ComputingService{
	ComputingServiceImpl () throws RemoteException  {
		super();
	}
	ArrayList<Book> all_book = new ArrayList<Book>();
	BookList query = new BookList();	

	

	public void initBook() throws RemoteException, FileNotFoundException {
		File file = new File("book.txt");
		Scanner input = new Scanner(file);	
		while (input.hasNext()) {
			int id = input.nextInt();
			String name = input.next();
			all_book.add(new Book(id, name));
		}
		input.close();
	}

	public void savetxt() throws RemoteException, FileNotFoundException {
		File file = new File("book.txt");
		PrintWriter output = new PrintWriter(file);
		for (int i = 0; i < all_book.size(); i++) {
			output.print(all_book.get(i).getID() + " ");
			output.println(all_book.get(i).getName());
		}
		output.close();
	}



	public boolean add(Book b) throws RemoteException, FileNotFoundException{
		for (int i = 0; i < all_book.size(); i++) {
			if (all_book.get(i).getID() == b.getID()) {
				return false;
			}
		}
		all_book.add(b);
		savetxt();
		return true;
	}

	public boolean delete(int bookID) throws RemoteException, FileNotFoundException {
		for (int i = 0; i < all_book.size(); i++) {
			if (all_book.get(i).getID() == bookID) {
				all_book.remove(all_book.get(i));  
				savetxt();
				return true;
			}
		}
		return false;
	}

	public Book queryByID(int bookID) throws RemoteException{
		Book bb = null;
		for (int i = 0; i < all_book.size(); i++) {
			if (all_book.get(i).getID() == bookID) {
				bb = all_book.get(i);
				return bb;
			}
		}
		return null;
	}


	public BookList queryByName(String name) throws RemoteException{
		for (int i = 0; i < all_book.size(); i++) {
			if(all_book.get(i).getName().indexOf(name)>=0){
                query.booklist.add(all_book.get(i));
            }			
		}
		return query;
	}
	 

	 
	public String booksInfo() throws RemoteException {
		String info = "";
		for (int i = 0; i < all_book.size(); i++) {
			info += ("id:" + all_book.get(i).getID() 
			+ " name:" + all_book.get(i).getName() + "\n");
		}
		return info;
	}
	 
	public void showAll() throws RemoteException {
		System.out.println(booksInfo());
	}


	 
}
