'use client';

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

// This matches the JSON response from your FastAPI
interface UserData {
  uid: number;
  user: string;
  type: string;
}

interface AuthContextType {
  user: UserData | null;
  login: (data: UserData) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserData | null>(null);

  // Load session from localStorage on startup
  useEffect(() => {
    const savedSession = localStorage.getItem('marine_user');
    if (savedSession) {
      setUser(JSON.parse(savedSession));
    }
  }, []);

  const login = (data: UserData) => {
    setUser(data);
    localStorage.setItem('marine_user', JSON.stringify(data));
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('marine_user');
  };

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used within AuthProvider");
  return context;
};