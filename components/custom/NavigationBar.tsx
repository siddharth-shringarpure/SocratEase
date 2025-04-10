"use client";

/**
 * @fileoverview A responsive navigation bar component that provides site-wide navigation,
 * featuring both desktop and mobile layouts with smooth animations.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect } from "react";
import { Menu, X } from "lucide-react";

const NAV_ITEMS = [
  { href: "/", label: "Home" },
  { href: "/practice", label: "Practice" },
  { href: "/camera", label: "Camera" },
  { href: "/analytics", label: "Analytics" },
  { href: "/recordings", label: "Recordings" },
];

const NAV_ANIMATION = {
  hidden: { opacity: 0, y: -20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1] },
  },
};

const MOBILE_MENU_VARIANTS = {
  closed: {
    opacity: 0,
    transition: { duration: 0.2, ease: "easeInOut" },
  },
  open: {
    opacity: 1,
    transition: { duration: 0.3, ease: "easeOut" },
  },
};

const MOBILE_LINK_VARIANTS = {
  closed: { opacity: 0, x: 50 },
  open: (index: number) => ({
    opacity: 1,
    x: 0,
    transition: {
      delay: index * 0.1,
      duration: 0.3,
      ease: "easeOut",
    },
  }),
};

/**
 * Renders a responsive navigation bar with animated transitions
 * @returns {JSX.Element} The navigation component
 */
export function NavigationBar(): JSX.Element {
  const currentPath = usePathname();
  const [isOpen, setIsOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  // Check if we're on mobile and update state accordingly
  useEffect(() => {
    const checkIfMobile = () => {
      setIsMobile(window.innerWidth < 768); // breakpoint for mobile view
    };

    checkIfMobile();
    window.addEventListener("resize", checkIfMobile);

    return () => window.removeEventListener("resize", checkIfMobile);
  }, []);

  // Reset mobile menu when route changes
  useEffect(() => {
    setIsOpen(false);
  }, [currentPath]);

  // prevent scrolling when mobile menu is open
  useEffect(() => {
    document.body.style.overflow = isOpen ? "hidden" : "unset";
    return () => {
      document.body.style.overflow = "unset";
    };
  }, [isOpen]);

  return (
    <>
      <motion.nav
        initial="hidden"
        animate="visible"
        variants={NAV_ANIMATION}
        className="fixed top-0 left-0 right-0 z-50 border-b bg-background/80 backdrop-blur-md"
      >
        <div className="container flex h-16 items-center px-4 select-none">
          <Link href="/" className="mr-6 flex items-center space-x-2">
            <span className="text-xl font-bold">SocratEase</span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex gap-6">
            {NAV_ITEMS.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "transition-colors hover:text-foreground/80",
                  currentPath === item.href
                    ? "text-foreground"
                    : "text-foreground/60"
                )}
              >
                {item.label}
              </Link>
            ))}
          </div>

          {/* hamburger menu for mobile */}
          <div className="md:hidden ml-auto">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="p-2 hover:bg-accent rounded-md"
              aria-label={isOpen ? "Close menu" : "Open menu"}
            >
              <motion.div
                animate={{ rotate: isOpen ? 90 : 0 }}
                transition={{ duration: 0.2 }}
              >
                {isOpen ? (
                  <X className="h-6 w-6" />
                ) : (
                  <Menu className="h-6 w-6" />
                )}
              </motion.div>
            </button>
          </div>
        </div>
      </motion.nav>

      {/* Mobile Menu Overlay */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial="closed"
            animate="open"
            exit="closed"
            variants={MOBILE_MENU_VARIANTS}
            className="fixed inset-0 z-40 md:hidden"
          >
            {/* Backdrop blur overlay */}
            <motion.div
              className="absolute inset-0 bg-background/80 backdrop-blur-md"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            />

            <div className="relative h-full pt-20 px-4 flex flex-col">
              <div className="flex flex-col space-y-4">
                {NAV_ITEMS.map((item, index) => (
                  <motion.div
                    key={item.href}
                    custom={index}
                    variants={MOBILE_LINK_VARIANTS}
                    initial="closed"
                    animate="open"
                    exit="closed"
                  >
                    <Link
                      href={item.href}
                      className={cn(
                        "block py-3 px-4 text-lg rounded-md transition-colors",
                        currentPath === item.href
                          ? "bg-accent text-foreground font-medium"
                          : "hover:bg-accent/50 text-foreground/60"
                      )}
                      onClick={() => setIsOpen(false)}
                    >
                      {item.label}
                    </Link>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
