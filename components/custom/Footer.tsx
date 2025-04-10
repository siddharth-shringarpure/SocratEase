"use client";

/**
 * @fileoverview Provides a responsive footer component with animated transitions and navigation links.
 * Implements consistent branding and legal requirements across the application.
 */

import Link from "next/link";
import { motion } from "framer-motion";

const FOOTER_LINKS = [
  { href: "/privacy", label: "Privacy Policy" },
  { href: "/terms", label: "Terms of Service" },
];

const FOOTER_ANIMATION = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.6,
      ease: [0.22, 1, 0.36, 1],
    },
  },
};

/**
 * Renders a responsive footer with animated transitions and navigation
 * @returns {JSX.Element} The footer component
 */
export function Footer(): JSX.Element {
  const currentYear = new Date().getFullYear();

  return (
    <motion.footer
      initial="hidden"
      animate="visible"
      variants={FOOTER_ANIMATION}
      className="border-t bg-background/80 backdrop-blur-md mt-auto py-8"
    >
      <div className="container flex flex-col items-center px-4 mx-auto">
        <div className="flex flex-col items-center space-y-6 md:flex-row md:space-y-0 md:space-x-6 md:justify-between w-full">
          {/* Logo and copyright */}
          <div className="flex flex-col items-center md:items-start space-y-2">
            <Link href="/" className="flex items-center space-x-2">
              <span className="text-lg font-semibold">SocratEase</span>
            </Link>
            <p className="text-sm text-foreground/60">
              © {currentYear} SocratEase. All rights reserved.
              {/* TODO: Do we need copyright? */}
            </p>
          </div>

          {/* Links */}
          <div className="flex flex-col md:flex-row items-center space-y-3 md:space-y-0 md:space-x-6">
            {FOOTER_LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="text-sm text-foreground/60 hover:text-foreground/80 transition-colors"
              >
                {link.label}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </motion.footer>
  );
}
