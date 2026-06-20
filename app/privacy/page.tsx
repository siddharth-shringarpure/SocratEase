"use client";

import {
  motion,
  AnimatePresence,
  useInView,
  useScroll,
  useSpring,
  useAnimationFrame,
} from "framer-motion";
import Head from "next/head";
import Link from "next/link";
import { useState, useRef, useEffect } from "react";
import { ArrowUp, ChevronDown, ChevronUp, ExternalLink } from "lucide-react";
import React from "react";
import * as Accordion from "@radix-ui/react-accordion"; // Import Radix Accordion

/**
 * Privacy Policy page component
 * @returns {JSX.Element} The privacy policy page
 */
export default function PrivacyPolicy(): JSX.Element {
  // State to track active section for table of contents
  const [activeSection, setActiveSection] = useState<string>("intro");

  // State to track if mouse is near sidebar for focus effect
  const [isMouseNearSidebar, setIsMouseNearSidebar] = useState<boolean>(false);

  // State to track if page has loaded
  const [pageLoaded, setPageLoaded] = useState<boolean>(false);

  // Refs for each section
  const sectionRefs = {
    intro: useRef<HTMLElement>(null),
    data: useRef<HTMLElement>(null),
    storage: useRef<HTMLElement>(null),
    rights: useRef<HTMLElement>(null),
    changes: useRef<HTMLElement>(null),
    contact: useRef<HTMLElement>(null),
  };

  // Main content ref for handling focus/blur effect
  const contentRef = useRef<HTMLDivElement>(null);

  // Remove isScrolling state as it's not needed
  const [targetScrollY, setTargetScrollY] = useState<number | null>(null);

  // Create a spring animation for smooth scrolling
  const springConfig = { stiffness: 80, damping: 20, mass: 1 };
  const scrollY = useSpring(0, springConfig);

  // Use animation frame to update scroll position
  useAnimationFrame(() => {
    if (targetScrollY !== null) {
      const currentScroll = window.scrollY;
      const diff = Math.abs(targetScrollY - currentScroll);

      if (diff < 1) {
        setTargetScrollY(null);
        return;
      }

      // Calculate a step towards the target with spring-like easing
      const step = (targetScrollY - currentScroll) * 0.1;
      window.scrollTo(0, currentScroll + step);
    }
  });

  // Update active section based on scroll position
  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY + 150;

      // Find the section that is currently in view
      let currentSection = activeSection;
      for (const sectionId of Object.keys(sectionRefs)) {
        const element =
          sectionRefs[sectionId as keyof typeof sectionRefs].current;
        if (element) {
          const { offsetTop, offsetHeight } = element;
          if (
            scrollPosition >= offsetTop &&
            scrollPosition < offsetTop + offsetHeight
          ) {
            currentSection = sectionId;
            break;
          }
        }
      }

      if (activeSection !== currentSection) {
        setActiveSection(currentSection);
      }
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, [activeSection, sectionRefs]);

  // Get current date for last updated
  const currentDate = new Date().toLocaleDateString("en-GB", {
    day: "numeric",
    month: "long",
    year: "numeric",
  });

  // Table of contents sections
  const sections = [
    { id: "intro", label: "Our Privacy Commitment" },
    { id: "data", label: "Data We Collect & Use" },
    { id: "storage", label: "How We Store Your Data" },
    { id: "rights", label: "Your Rights" },
    { id: "changes", label: "Policy Updates" },
    { id: "contact", label: "Contact Us" },
  ];

  // Update scroll to section function
  const scrollToSection = (sectionId: string) => {
    const element = sectionRefs[sectionId as keyof typeof sectionRefs].current;
    if (element) {
      const headerOffsetTop = -72;
      const targetOffsetTop = element.offsetTop;
      const scrollPosition = targetOffsetTop - headerOffsetTop;

      setTargetScrollY(scrollPosition);
      scrollY.set(scrollPosition);
      setActiveSection(sectionId);
    }
  };

  // Update scroll to top function with smooth animation
  const scrollToTop = () => {
    const currentScroll = window.scrollY;
    setTargetScrollY(0);
    scrollY.set(currentScroll);
    // Don't immediately set to 0, let the spring animation work naturally
  };

  // Reusable Radix Accordion Item Component
  const AccordionItem = React.forwardRef<
    HTMLDivElement,
    Accordion.AccordionItemProps & {
      children: React.ReactNode;
      className?: string;
    }
  >(({ children, className, ...props }, forwardedRef) => (
    <Accordion.Item
      className={`my-6 border border-border rounded-md overflow-hidden ${className}`}
      {...props}
      ref={forwardedRef}
    >
      {children}
    </Accordion.Item>
  ));
  AccordionItem.displayName = "AccordionItem";

  const AccordionTrigger = React.forwardRef<
    HTMLButtonElement,
    Accordion.AccordionTriggerProps & {
      children: React.ReactNode;
      className?: string;
    }
  >(({ children, className, ...props }, forwardedRef) => (
    <Accordion.Header className="flex">
      <Accordion.Trigger
        className={`group w-full p-4 flex justify-between items-center text-left bg-secondary/5 
        hover:bg-secondary/10 transition-colors data-[state=open]:bg-secondary/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ${className}`}
        {...props}
        ref={forwardedRef}
      >
        <span className="font-semibold">{children}</span>
        <ChevronDown
          size={18}
          className="transition-transform duration-300 ease-[cubic-bezier(0.87,_0,_0.13,_1)] group-data-[state=open]:rotate-180"
          aria-hidden
        />
      </Accordion.Trigger>
    </Accordion.Header>
  ));
  AccordionTrigger.displayName = "AccordionTrigger";

  const AccordionContent = React.forwardRef<
    HTMLDivElement,
    Accordion.AccordionContentProps & {
      children: React.ReactNode;
      className?: string;
    }
  >(({ children, className, ...props }, forwardedRef) => (
    <Accordion.Content
      className={`data-[state=open]:animate-slideDown data-[state=closed]:animate-slideUp overflow-hidden relative z-30 ${className}`}
      {...props}
      ref={forwardedRef}
    >
      <div className="p-6 bg-background text-foreground/80 text-sm">
        {children}
      </div>
    </Accordion.Content>
  ));
  AccordionContent.displayName = "AccordionContent";

  // Section with blur effect that fades as it comes into view
  // Now also aware of sidebar hover to prevent conflicting blurs
  const BlurFadeSection = ({ children }: { children: React.ReactNode }) => {
    const ref = useRef(null);
    const isInView = useInView(ref, {
      margin: "-10% 0px -10% 0px",
      amount: 0.3,
    });

    return (
      <motion.div
        ref={ref}
        animate={{
          filter: isInView ? "blur(0px)" : "blur(5px)",
          opacity: isInView ? 1 : 0.7,
        }}
        transition={{ duration: 0.6, ease: "easeOut" }}
      >
        {children}
      </motion.div>
    );
  };

  // Policy Section component
  const PolicySection = ({
    id,
    title,
    children,
  }: {
    id: string;
    title: string;
    children: React.ReactNode;
  }) => {
    const ref = sectionRefs[id as keyof typeof sectionRefs];

    return (
      <section
        id={id}
        ref={ref}
        className="mb-36 last:mb-0 scroll-mt-24 relative"
      >
        {/* Sticky header implementation */}
        <div className="sticky top-[72px] z-20">
          {/* Shield div using padding instead of negative positioning */}
          <div className="absolute inset-x-0 top-0 bg-background z-10">
            <div className="pt-16 pb-4" />
          </div>

          {/* Header content with higher z-index */}
          <div className="bg-background/95 backdrop-blur-sm relative z-20">
            <div className="py-6">
              <h2 className="text-3xl font-semibold">{title}</h2>
            </div>
            {/* Gradient fade for smooth transition */}
            <div className="absolute -bottom-6 left-0 right-0 h-6 bg-gradient-to-b from-background/95 to-transparent pointer-events-none" />
          </div>
        </div>

        {/* Content container with consistent spacing */}
        <div className="pt-24 pb-16 space-y-10 relative z-10">
          <BlurFadeSection>{children}</BlurFadeSection>
        </div>
      </section>
    );
  };

  // Animation variants for staggered animations
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15,
        delayChildren: 0.3,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.7,
        ease: [0.25, 0.1, 0.25, 1.0],
      },
    },
  };

  // Animation variants for sidebar elements
  const sidebarContainerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.08,
        delayChildren: 0.5,
      },
    },
  };

  const sidebarItemVariants = {
    hidden: { opacity: 0, x: -20 },
    visible: {
      opacity: 1,
      x: 0,
      transition: {
        duration: 0.5,
        ease: [0.25, 0.1, 0.25, 1.0],
      },
    },
  };

  // Effect to set page as loaded after a short delay
  useEffect(() => {
    const timer = setTimeout(() => {
      setPageLoaded(true);
    }, 100);

    return () => clearTimeout(timer);
  }, []);

  return (
    <>
      <Head>
        <title>Privacy Policy - SocratEase</title>
        <meta
          name="description"
          content="SocratEase Privacy Policy and GDPR Compliance Information"
        />
      </Head>

      <div className="bg-background min-h-screen pb-32">
        {/* Fixed back to top button */}
        <motion.button
          onClick={scrollToTop}
          className="fixed bottom-8 right-8 z-50 p-3 bg-background/80 backdrop-blur-sm border border-border rounded-full shadow-lg hover:bg-background/95 transition-all group"
          aria-label="Back to top"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: pageLoaded ? 1 : 0, scale: pageLoaded ? 1 : 0.8 }}
          whileHover={{ scale: 1.1 }}
          transition={{ duration: 0.2 }}
        >
          <ArrowUp className="h-5 w-5 text-foreground group-hover:-translate-y-0.5 transition-transform" />
        </motion.button>

        <div className="container mx-auto px-6 pt-32 pb-16 relative">
          {/* Blur Overlay - Positioned before content */}
          <motion.div
            className="fixed inset-0 pointer-events-none z-10"
            initial={{ opacity: 0 }}
            animate={{
              opacity: isMouseNearSidebar ? 1 : 0,
            }}
            transition={{ duration: 0.4 }}
            style={{
              backdropFilter: isMouseNearSidebar ? "blur(4px)" : "blur(0px)",
              background: isMouseNearSidebar
                ? "rgba(0, 0, 0, 0.2)"
                : "rgba(0, 0, 0, 0)",
              transition: "backdrop-filter 0.4s ease, background 0.4s ease",
            }}
          />

          {/* Sidebar navigation - Further away from content */}
          <div
            className="hidden lg:block fixed left-10 xl:left-20 2xl:left-40 top-1/2 -translate-y-1/2 z-30"
            onMouseEnter={() => setIsMouseNearSidebar(true)}
            onMouseLeave={() => setIsMouseNearSidebar(false)}
          >
            <motion.div
              className="flex flex-col space-y-6 items-center relative z-10 p-4 -m-4"
              variants={sidebarContainerVariants}
              initial="hidden"
              animate="visible"
              whileHover={{
                opacity: isMouseNearSidebar ? 1 : 0.7,
                scale: isMouseNearSidebar ? 1.05 : 1,
              }}
              transition={{ duration: 0.3 }}
            >
              {sections.map((section) => (
                <motion.button
                  key={section.id}
                  onClick={() => scrollToSection(section.id)}
                  className="group relative flex items-center w-full"
                  aria-label={section.label}
                  variants={sidebarItemVariants}
                >
                  <motion.div
                    className={`w-3 h-3 rounded-full transition-colors duration-300 ${
                      activeSection === section.id
                        ? "bg-primary"
                        : "bg-secondary/40 group-hover:bg-secondary/70"
                    }`}
                    whileHover={{ scale: 1.2 }}
                    animate={{
                      scale: activeSection === section.id ? 1.2 : 1,
                      boxShadow:
                        activeSection === section.id
                          ? "0 0 8px rgba(17, 17, 27, 0.5)"
                          : "none",
                    }}
                  ></motion.div>
                  <motion.span
                    className={`absolute left-6 py-2 px-4 text-xs rounded whitespace-nowrap bg-background/80 backdrop-blur-sm min-h-[32px] flex items-center ${
                      activeSection === section.id
                        ? "text-primary"
                        : "text-foreground/70"
                    }`}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{
                      opacity:
                        isMouseNearSidebar || activeSection === section.id
                          ? 1
                          : 0,
                      x:
                        isMouseNearSidebar || activeSection === section.id
                          ? 0
                          : -10,
                    }}
                    transition={{ duration: 0.2 }}
                  >
                    {section.label}
                  </motion.span>
                </motion.button>
              ))}
            </motion.div>
          </div>

          {/* Main content area - Centered with proper max width */}
          <div className="max-w-3xl mx-auto w-full relative">
            {/* Actual Content Container */}
            <div ref={contentRef} className="relative z-10">
              {/* Title section */}
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, ease: "easeOut" }}
                className="text-center mb-40"
              >
                <h1 className="text-4xl md:text-5xl font-bold mb-4">
                  Privacy Policy
                </h1>
                <p className="text-muted-foreground">
                  Last updated: {currentDate}
                </p>
              </motion.div>

              {/* Content Sections Container - Use Radix Accordion Root here */}
              <motion.div
                variants={containerVariants}
                initial="hidden"
                animate={pageLoaded ? "visible" : "hidden"}
              >
                <Accordion.Root
                  type="multiple" // Allow multiple items open at once
                  className="space-y-0" // PolicySection handles margin
                >
                  <motion.div variants={itemVariants}>
                    <PolicySection id="intro" title="Our Privacy Commitment">
                      {/* Intro content (no accordion here) */}
                      <div className="bg-gradient-to-br from-secondary/5 to-primary/5 p-8 rounded-lg shadow-sm">
                        <p className="text-xl leading-relaxed">
                          At SocratEase, we prioritise your privacy. We collect
                          minimal data and delete it as soon as we can. We
                          don&apos;t have user accounts, we don&apos;t use
                          cookies, and we don&apos;t sell your data.
                        </p>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-10 mt-16">
                        <div>
                          <div className="flex items-center mb-4">
                            <span className="text-3xl mr-3">🔍</span>
                            <h3 className="text-xl font-semibold">
                              Minimal Collection
                            </h3>
                          </div>
                          <p className="text-foreground/80 leading-relaxed">
                            We only collect what&apos;s necessary to provide our
                            service — video and audio when you practice, and
                            basic usage data.
                          </p>
                        </div>
                        <div>
                          <div className="flex items-center mb-4">
                            <span className="text-3xl mr-3">⏱️</span>
                            <h3 className="text-xl font-semibold">
                              Brief Storage
                            </h3>
                          </div>
                          <p className="text-foreground/80 leading-relaxed">
                            Media files are only kept for as long as needed for
                            processing, then they're deleted.
                          </p>
                        </div>
                      </div>

                      <div className="mt-16 bg-primary/5 p-8 rounded-lg border border-primary/20 shadow-sm">
                        <div className="flex items-center mb-4">
                          <span className="text-3xl mr-3">🔒</span>
                          <h3 className="text-xl font-semibold text-primary">
                            We Don't Train AI on Your Data
                          </h3>
                        </div>
                        <p className="text-foreground/90 leading-relaxed">
                          Unlike many AI platforms, we <strong>never</strong>{" "}
                          use your recordings, speech, or practice sessions to
                          train our AI models. Your content is only used to
                          provide you with immediate feedback, and then it's
                          deleted. We believe your privacy and intellectual
                          property should be respected, not used as training
                          material.
                        </p>
                      </div>
                    </PolicySection>
                  </motion.div>

                  <motion.div variants={itemVariants}>
                    <PolicySection id="data" title="Data We Collect & Use">
                      {/* ... Data section intro ... */}
                      <div className="bg-gradient-to-br from-secondary/5 to-primary/5 p-8 rounded-lg shadow-sm">
                        <p className="text-xl leading-relaxed">
                          We collect video and audio during practice sessions to
                          give you feedback. Media is deleted after analysis.
                          Your settings are saved on your device, not our
                          servers.
                        </p>
                      </div>
                      <div className="mt-16 space-y-10">
                        {/* ... Video/audio and settings paragraphs ... */}
                        <div>
                          <div className="flex items-center mb-4">
                            <span className="text-3xl mr-3">🎥</span>
                            <h3 className="text-xl font-semibold">
                              Video and audio recordings
                            </h3>
                          </div>
                          <p className="text-foreground/80 leading-relaxed">
                            Only collected when you explicitly initiate practice
                            sessions. These recordings are used for analysis to
                            provide feedback and are deleted once processing is
                            complete.
                          </p>
                        </div>
                        <div>
                          <div className="flex items-center mb-4">
                            <span className="text-3xl mr-3">📱</span>
                            <h3 className="text-xl font-semibold">
                              Your settings and preferences
                            </h3>
                          </div>
                          <p className="text-foreground/80 leading-relaxed">
                            How you use the app and your preferences are stored
                            directly on your device rather than on our servers.
                          </p>
                        </div>
                        <div>
                          <div className="flex items-center mb-4">
                            <span className="text-3xl mr-3">🤖</span>
                            <h3 className="text-xl font-semibold">
                              No AI training on your content
                            </h3>
                          </div>
                          <p className="text-foreground/80 leading-relaxed">
                            We use AI to provide feedback on your practice
                            sessions, but we don't use your data to train or
                            improve our AI models. Your content remains private
                            and is never incorporated into our training
                            datasets.
                          </p>
                        </div>
                        {/* Replace DetailDisclosure with Radix Accordion components */}
                        <AccordionItem value="technical-data">
                          <AccordionTrigger>
                            Technical details about the data we collect
                          </AccordionTrigger>
                          <AccordionContent>
                            {/* ... Technical data content ... */}
                            <div className="space-y-4">
                              <div>
                                <h4 className="font-semibold mb-2">
                                  Basic technical data
                                </h4>
                                <p>
                                  Device type and IP address for service
                                  delivery and security. This helps us ensure
                                  the app works correctly on your device.
                                </p>
                              </div>
                              <div>
                                <h4 className="font-semibold mb-2">
                                  Feedback content
                                </h4>
                                <p>
                                  Text from your practice sessions that we use
                                  to generate feedback and improvement
                                  suggestions.
                                </p>
                              </div>
                              <div>
                                <h4 className="font-semibold mb-2">
                                  Third-party services
                                </h4>
                                <p>
                                  We connect with Neuphonic API to enhance your
                                  experience, but we only send generated
                                  feedback text (never any PII). Speech
                                  processing happens locally on our backend.
                                </p>
                                <p className="mt-2">
                                  Our app is hosted on Vercel and Render, which
                                  may collect basic analytics about visits to
                                  our site as part of their platform services.
                                </p>
                              </div>
                            </div>
                          </AccordionContent>
                        </AccordionItem>

                        <AccordionItem value="legal-basis">
                          <AccordionTrigger>
                            Legal basis for data processing (GDPR details)
                          </AccordionTrigger>
                          <AccordionContent>
                            {/* ... Legal basis content ... */}
                            <div>
                              <p>
                                Under the General Data Protection Regulation, we
                                process your data based on:
                              </p>
                              <ul className="list-disc pl-6 mt-4 space-y-2">
                                <li>
                                  Your consent — when you choose to use our
                                  service
                                </li>
                                <li>
                                  Legitimate interests — to improve our service
                                  and ensure security
                                </li>
                              </ul>
                            </div>
                          </AccordionContent>
                        </AccordionItem>
                      </div>
                    </PolicySection>
                  </motion.div>

                  <motion.div variants={itemVariants}>
                    <PolicySection id="storage" title="How We Store Your Data">
                      {/* ... Storage section intro ... */}
                      <div className="bg-gradient-to-br from-secondary/5 to-primary/5 p-8 rounded-lg shadow-sm">
                        <p className="text-xl leading-relaxed">
                          Your media files are only kept for the time needed to
                          process them. Your preferences are saved on your
                          device.
                        </p>
                      </div>
                      <div className="mt-16 space-y-10">
                        {/* ... Short-term processing paragraph ... */}
                        <div>
                          <div className="flex items-center mb-4">
                            <span className="text-3xl mr-3">⏳</span>
                            <h3 className="text-xl font-semibold">
                              Short-term processing
                            </h3>
                          </div>
                          <p className="text-foreground/80 leading-relaxed">
                            Video and audio recordings are only kept for the
                            time needed to process them and provide feedback.
                            They're deleted once the analysis is finished.
                          </p>
                        </div>
                        <AccordionItem value="data-retention">
                          <AccordionTrigger>
                            Technical details about data handling and security
                          </AccordionTrigger>
                          <AccordionContent>
                            {/* ... Data retention content ... */}
                            <div className="space-y-4">
                              <div>
                                <h4 className="font-semibold mb-2">
                                  Basic safeguards
                                </h4>
                                <p>
                                  We use standard safeguards and follow basic
                                  security practices when handling your data.
                                  However, video and audio sent to our backend
                                  are not specifically encrypted beyond standard
                                  HTTPS transmission.
                                </p>
                              </div>
                              <div>
                                <h4 className="font-semibold mb-2">
                                  Data Retention
                                </h4>
                                <p>
                                  We only keep data for as long as necessary:
                                </p>
                                <ul className="list-disc pl-6 mt-2 space-y-2">
                                  <li>
                                    Media recordings are deleted after
                                    processing is complete
                                  </li>
                                  <li>
                                    Settings remain on your device until you
                                    clear your browser data
                                  </li>
                                </ul>
                              </div>
                              <div>
                                <h4 className="font-semibold mb-2">
                                  Recovery period
                                </h4>
                                <p>
                                  If processing fails, files may remain for up
                                  to an hour before being automatically deleted.
                                </p>
                              </div>
                            </div>
                          </AccordionContent>
                        </AccordionItem>
                      </div>
                    </PolicySection>
                  </motion.div>

                  <motion.div variants={itemVariants}>
                    <PolicySection id="rights" title="Your Rights">
                      {/* ... Rights section intro ... */}
                      <div className="bg-gradient-to-br from-secondary/5 to-primary/5 p-8 rounded-lg shadow-sm">
                        <p className="text-xl leading-relaxed">
                          You have the right to access, correct, or delete your
                          data. You can also object to how we use it. Since we
                          don&apos;t maintain user accounts, many of these
                          rights can be exercised directly through the app or by
                          contacting us.
                        </p>
                      </div>
                      <p className="mt-10 leading-relaxed">
                        To exercise these rights, please contact us using the
                        details in the "Contact Us" section. Since we don&apos;t
                        maintain user accounts, we have limited stored data, but
                        we will respond to all legitimate requests within 30
                        days.
                      </p>
                      <div className="mt-8">
                        <AccordionItem value="gdpr-rights">
                          <AccordionTrigger>
                            Full details on your data protection rights
                          </AccordionTrigger>
                          <AccordionContent>
                            {/* ... GDPR rights content ... */}
                            <div>
                              <p className="mb-4">
                                Under the GDPR, you have the following rights:
                              </p>
                              <ul className="list-disc pl-6 space-y-3">
                                <li>
                                  <strong>Right to Access</strong> — Request
                                  information about what data we have about you
                                </li>
                                <li>
                                  <strong>Right to Rectification</strong> —
                                  Request correction of inaccurate information
                                </li>
                                <li>
                                  <strong>Right to Erasure</strong> — Request
                                  deletion of your data (the "right to be
                                  forgotten")
                                </li>
                                <li>
                                  <strong>Right to Object</strong> — Object to
                                  the processing of your personal data
                                </li>
                                <li>
                                  <strong>Right to Restrict Processing</strong>{" "}
                                  — Request limits on how we use your data
                                </li>
                                <li>
                                  <strong>Right to Data Portability</strong> —
                                  Request your data in a machine-readable format
                                </li>
                                <li>
                                  <strong>Right to Withdraw Consent</strong> —
                                  Revoke previously given consent at any time
                                </li>
                                <li>
                                  <strong>Right to Complain</strong> — Lodge a
                                  complaint with a data protection authority
                                </li>
                              </ul>
                            </div>
                          </AccordionContent>
                        </AccordionItem>
                      </div>
                    </PolicySection>
                  </motion.div>

                  <motion.div variants={itemVariants}>
                    <PolicySection id="changes" title="Policy Updates">
                      {/* ... Changes section content ... */}
                      <div className="bg-gradient-to-br from-secondary/5 to-primary/5 p-8 rounded-lg shadow-sm">
                        <p className="text-xl leading-relaxed">
                          We may update this policy from time to time. The
                          latest version will be available on this page.
                        </p>
                      </div>

                      <div className="mt-10">
                        <p className="text-foreground/80 leading-relaxed">
                          Last updated on {currentDate}. Significant changes to
                          this policy may be communicated through notices on our
                          website.
                        </p>
                      </div>
                    </PolicySection>
                  </motion.div>

                  <motion.div variants={itemVariants}>
                    <PolicySection id="contact" title="Contact Us">
                      {/* ... Contact section intro ... */}
                      <div className="bg-gradient-to-br from-secondary/5 to-primary/5 p-8 rounded-lg shadow-sm">
                        <div className="flex flex-col md:flex-row md:items-center justify-between">
                          <div>
                            <p className="text-xl mb-4">
                              Questions about your privacy?
                            </p>
                            <p className="text-foreground/80">
                              <strong>Email:</strong> hello.socratease@proton.me
                            </p>
                          </div>
                          <div className="mt-6 md:mt-0">
                            <Link
                              href="https://ico.org.uk"
                              target="_blank"
                              rel="noopener noreferrer"
                              className="flex items-center text-primary hover:underline"
                            >
                              UK Information Commissioner&apos;s Office
                              <ExternalLink size={16} className="ml-1" />
                            </Link>
                          </div>
                        </div>
                      </div>
                      <div className="mt-10">
                        <AccordionItem value="contact-details">
                          <AccordionTrigger>
                            Detailed contact information
                          </AccordionTrigger>
                          <AccordionContent>
                            {/* ... Contact details content ... */}
                            <div className="space-y-4">
                              <div>
                                <h4 className="font-semibold mb-2">
                                  Our contact details:
                                </h4>
                                <p className="mb-2">
                                  <strong>Email:</strong>{" "}
                                  hello.socratease@proton.me
                                </p>
                              </div>
                              <div>
                                <h4 className="font-semibold mb-2">
                                  UK Supervisory Authority
                                </h4>
                                <p className="mb-2">
                                  <strong>
                                    Information Commissioner&apos;s Office (ICO)
                                  </strong>
                                </p>
                                <p className="mb-2">
                                  <strong>Website:</strong>{" "}
                                  <a
                                    href="https://ico.org.uk"
                                    className="text-blue-400 hover:underline"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                  >
                                    https://ico.org.uk
                                  </a>
                                </p>
                                <p className="mb-2">
                                  <strong>Phone:</strong> 0303 123 1113
                                </p>
                                <p>
                                  <strong>Address:</strong> Information
                                  Commissioner&apos;s Office, Wycliffe House,
                                  Water Lane, Wilmslow, Cheshire, SK9 5AF
                                </p>
                              </div>
                            </div>
                          </AccordionContent>
                        </AccordionItem>
                      </div>
                    </PolicySection>
                  </motion.div>

                  {/* Final statement */}
                  <motion.div
                    variants={itemVariants}
                    className="border-t border-border pt-8 mt-16 text-center"
                  >
                    <p className="text-muted-foreground">
                      Thank you for taking the time to understand how we handle
                      your data. We&apos;re committed to protecting your
                      privacy.
                    </p>
                  </motion.div>
                </Accordion.Root>
              </motion.div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
