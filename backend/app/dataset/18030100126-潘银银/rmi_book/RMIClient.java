import java.io.*;
import java.util.*;
import java.net.MalformedURLException;
import java.rmi.Naming;
import java.rmi.NotBoundException;
import java.rmi.RemoteException;
import java.io.Serializable;


public class RMIClient {
	
    public static void main(String[] args) throws RemoteException, NotBoundException, MalformedURLException, FileNotFoundException {     
        ComputingService computingObj = (ComputingService)Naming.lookup("rmi://localhost:8889/ComputingService");
		computingObj.initBook();
		Scanner input = new Scanner(System.in);
			System.out.println("----Book Manager System----\n"+"0.exit\n"+"1.add book\n"+"2.query Book by ID\n"+ "3.query Book by keyword\n"+ "4.delete book\n"+ "5.show all books\n"+ "---------------------------");		int cas = input.nextInt();
		while (cas != 0) {
			switch(cas) {
				case 1: System.out.println("Please input the id and name of the newBook:");
						int id = input.nextInt();
						String name = input.next();
						Book newBook = new Book(id, name);
						if (computingObj.add(newBook)) {
							System.out.println("-----add successful-----");
						} else {
							System.out.println("-----add failed-----");
						}
						break;
				case 2: System.out.println("Please input the id  of the Book that you want to find:");
						int queryBookID = input.nextInt();
						Book queryid = computingObj.queryByID(queryBookID);
						if (queryid != null) {
							System.out.println("-------results-------");
							queryid.showInfo();
						} else {
							System.out.println("------not found------");
						}
						break;
				case 3: System.out.println("Please input the keyword of the Book that you want to find:");
						String pattern = input.next();
						BookList list = computingObj.queryByName(pattern);
						if (list != null) {
							System.out.println("-------results-------");
							list.showInfo();
						} else {
							System.out.println("------not found------");
						}
						break;
				case 4: System.out.println("Please input the keyword of the Book that you want to delete:");
						int deleteID = input.nextInt();
						if (computingObj.delete(deleteID)) {
							System.out.println("-----delete successful-----");
						} else {
							System.out.println("------delete failed------");
						}
						break;
				case 5: System.out.println("-------all books-------");
						System.out.println(computingObj.booksInfo());
						break;
				default: System.out.println("----please input the instruction----");
						break;
			}
			System.out.println("----Book Manager System----\n"+"0.exit\n"+"1.add book\n"+"2.query Book by ID\n"+ "3.query Book by keyword\n"+ "4.delete book\n"+ "5.show all books\n"+ "---------------------------");
			cas = input.nextInt();
		}
	}
}